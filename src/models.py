"""Syllabus model zoo (1-D adaptations for multichannel time series).

Every encoder maps one modality window (B, C, T) to an embedding (B, d).
Only architectures named in the course syllabus are used:

  Deep feed-forward network ............ MLP (BatchNorm / WeightNorm)
  CNN architectures .................... VGG, Inception, ResNet, Dilated ResNet, DenseNet,
                                         MobileNet, EfficientNet, ConvNeXt, Conv-AE
  Recurrent models ..................... RNN, GRU, LSTM (bidirectional, many-to-one,
                                         attention pooling), unidirectional LSTM, AE + LSTM
  Attention models ..................... Transformer encoder, Vision Transformer (ViT), BERT,
                                         Multi-task Transformer
  Deep generative / unsupervised ....... Autoencoder (basic / denoising / sparse / conv_ae), VAE,
                                         BiGAN, Conditional GAN (cGAN), BERT masked-token modelling

1-D adaptation: Conv2d -> Conv1d, 2-D patches -> 1-D time patches; architecture-defining
ideas (VGG 3-tap stacks, Inception branches, residual/dense connections, depthwise-separable
convs, MBConv+SE, ConvNeXt blocks, [CLS]/patch tokens) are kept unchanged.
Multimodal fusion uses a Transformer encoder over the per-modality embeddings.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")      # He initialisation
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def cbr(c_in, c_out, k=3, s=1, g=1, act=nn.ReLU):
    """Conv -> BatchNorm -> activation."""
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, s, k // 2, groups=g, bias=False),
                         nn.BatchNorm1d(c_out), act())


def adaptive_avg_pool1d(x, out_len: int):
    """MPS-safe 1-D adaptive average pooling (Apple Silicon MPS requires input % output == 0)."""
    if x.shape[-1] == out_len:
        return x
    if x.device.type == "mps" and (x.shape[-1] % out_len != 0):
        return F.adaptive_avg_pool1d(x.cpu(), out_len).to(x.device)
    return F.adaptive_avg_pool1d(x, out_len)


class Head(nn.Module):
    """Global average pooling + linear projection to the embedding size."""

    def __init__(self, c, d):
        super().__init__()
        self.fc = nn.Linear(c, d)

    def forward(self, x):
        return self.fc(x.mean(-1))


# =============================================================== deep feed-forward
class MLP(nn.Module):
    """Deep feed-forward network. norm="bn" (Batch Norm) or "wn" (Weight Normalization)."""

    def __init__(self, c_in, d=128, steps=32, p=0.3, norm="bn"):
        super().__init__()
        self.steps = steps
        wn = nn.utils.parametrizations.weight_norm

        def fc(i, o):
            if norm == "wn":
                return [wn(nn.Linear(i, o)), nn.ReLU(), nn.Dropout(p)]
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p)]
        self.net = nn.Sequential(nn.Flatten(), *fc(c_in * steps, 512), *fc(512, 256),
                                 nn.Linear(256, d))

    def forward(self, x):
        return self.net(adaptive_avg_pool1d(x, self.steps))


# =============================================================== VGG
class VGG(nn.Module):
    """Stacks of 3-tap convs + max-pooling (VGG-style), doubling width after each pool."""
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 256]

    def __init__(self, c_in, d=128):
        super().__init__()
        layers, c = [], c_in
        for v in self.cfg:
            if v == "M":
                layers.append(nn.MaxPool1d(2))
            else:
                layers.append(cbr(c, v, 3))
                c = v
        self.features = nn.Sequential(*layers)
        self.head = Head(c, d)

    def forward(self, x):
        return self.head(self.features(x))


# =============================================================== Inception
class InceptionModule(nn.Module):
    """Parallel 1x1, 1x1->3, 1x1->5 and pool->1x1 branches, concatenated (GoogLeNet)."""

    def __init__(self, c_in, b1, b3r, b3, b5r, b5, bp):
        super().__init__()
        self.b1 = cbr(c_in, b1, 1)
        self.b3 = nn.Sequential(cbr(c_in, b3r, 1), cbr(b3r, b3, 3))
        self.b5 = nn.Sequential(cbr(c_in, b5r, 1), cbr(b5r, b5, 5))
        self.bp = nn.Sequential(nn.MaxPool1d(3, 1, 1), cbr(c_in, bp, 1))
        self.out = b1 + b3 + b5 + bp

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)


class Inception(nn.Module):
    def __init__(self, c_in, d=128):
        super().__init__()
        self.stem = cbr(c_in, 64, 7)
        m1 = InceptionModule(64, 32, 32, 48, 8, 24, 24)
        m2 = InceptionModule(m1.out, 48, 48, 64, 12, 32, 32)
        m3 = InceptionModule(m2.out, 64, 64, 96, 16, 48, 48)
        self.body = nn.Sequential(m1, nn.MaxPool1d(2), m2, nn.MaxPool1d(2), m3)
        self.head = Head(m3.out, d)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# =============================================================== ResNet
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, s=1, dil=1):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(c_in, c_out, 3, s, dil, dilation=dil, bias=False),
                                nn.BatchNorm1d(c_out), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv1d(c_out, c_out, 3, 1, dil, dilation=dil, bias=False),
                                nn.BatchNorm1d(c_out))
        self.skip = (nn.Identity() if s == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, s, bias=False), nn.BatchNorm1d(c_out)))

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.skip(x))


class ResNet(nn.Module):
    """ResNet-18-style: stem + 4 stages of 2 basic residual blocks.
    dilated=True -> Dilated Residual Network: dilation 1,2,4,8 per stage instead of striding
    after stage 2, enlarging the receptive field without losing temporal resolution."""

    def __init__(self, c_in, d=128, widths=(64, 64, 128, 128), dilated=False):
        super().__init__()
        self.stem = cbr(c_in, widths[0], 7)
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            s = 1 if i == 0 or (dilated and i > 1) else 2
            dil = 2 ** i if dilated else 1
            blocks += [BasicBlock(c, w, s, dil), BasicBlock(w, w, 1, dil)]
            c = w
        self.body = nn.Sequential(*blocks)
        self.head = Head(c, d)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# =============================================================== DenseNet
class DenseLayer(nn.Module):
    def __init__(self, c_in, g):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm1d(c_in), nn.ReLU(), nn.Conv1d(c_in, 4 * g, 1, bias=False),
                               nn.BatchNorm1d(4 * g), nn.ReLU(), nn.Conv1d(4 * g, g, 3, 1, 1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.f(x)], 1)          # dense connectivity: concatenate features


class DenseNet(nn.Module):
    """3 dense blocks (growth rate 32, bottleneck BN-ReLU-1x1-BN-ReLU-3) with transition layers."""

    def __init__(self, c_in, d=128, g=32, layers=(4, 4, 4)):
        super().__init__()
        c = 64
        mods = [cbr(c_in, c, 7)]
        for i, n in enumerate(layers):
            for _ in range(n):
                mods.append(DenseLayer(c, g))
                c += g
            if i < len(layers) - 1:                    # transition: compress 0.5 + avg-pool
                mods += [nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c // 2, 1, bias=False),
                         nn.AvgPool1d(2)]
                c //= 2
        mods += [nn.BatchNorm1d(c), nn.ReLU()]
        self.body = nn.Sequential(*mods)
        self.head = Head(c, d)

    def forward(self, x):
        return self.head(self.body(x))


# =============================================================== MobileNet
def ds_conv(c_in, c_out, s=1, k=3):
    """Depthwise-separable convolution (MobileNet-v1): depthwise k-tap + pointwise 1x1."""
    return nn.Sequential(cbr(c_in, c_in, k, s, g=c_in), cbr(c_in, c_out, 1))


class MobileNet(nn.Module):
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (256, 2), (256, 1)]

    def __init__(self, c_in, d=128):
        super().__init__()
        layers, c = [cbr(c_in, 32, 3)], 32
        for w, s in self.cfg:
            layers.append(ds_conv(c, w, s))
            c = w
        self.body = nn.Sequential(*layers)
        self.head = Head(c, d)

    def forward(self, x):
        return self.head(self.body(x))


# =============================================================== EfficientNet
class SqueezeExcite(nn.Module):
    def __init__(self, c, c_red):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(c, c_red), nn.SiLU(), nn.Linear(c_red, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x.mean(-1)).unsqueeze(-1)


class MBConv(nn.Module):
    """Inverted residual: 1x1 expand -> depthwise k -> SE -> 1x1 project (+ skip)."""

    def __init__(self, c_in, c_out, expand, k, s):
        super().__init__()
        h = c_in * expand
        layers = [cbr(c_in, h, 1, act=nn.SiLU)] if expand != 1 else []
        layers += [cbr(h, h, k, s, g=h, act=nn.SiLU), SqueezeExcite(h, max(4, c_in // 4)),
                   nn.Conv1d(h, c_out, 1, bias=False), nn.BatchNorm1d(c_out)]
        self.f = nn.Sequential(*layers)
        self.res = s == 1 and c_in == c_out

    def forward(self, x):
        return x + self.f(x) if self.res else self.f(x)


class EfficientNet(nn.Module):
    """EfficientNet-B0 stage layout, width-scaled down for 1-D signals."""
    cfg = [(1, 16, 1, 3, 1), (6, 24, 2, 3, 2), (6, 40, 2, 5, 2), (6, 80, 2, 3, 2), (6, 112, 1, 5, 1)]

    def __init__(self, c_in, d=128):
        super().__init__()
        layers, c = [cbr(c_in, 32, 3, act=nn.SiLU)], 32
        for e, w, n, k, s in self.cfg:
            for i in range(n):
                layers.append(MBConv(c, w, e, k, s if i == 0 else 1))
                c = w
        layers.append(cbr(c, 256, 1, act=nn.SiLU))
        self.body = nn.Sequential(*layers)
        self.head = Head(256, d)

    def forward(self, x):
        return self.head(self.body(x))


# =============================================================== ConvNeXt
class ChannelLN(nn.Module):
    """LayerNorm over channels for (B, C, T) tensors."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    """depthwise 7-tap conv -> LN -> 1x1 (4x) -> GELU -> 1x1 -> layer-scale -> residual."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv1d(c, c, 7, padding=3, groups=c)
        self.ln = nn.LayerNorm(c)
        self.pw1, self.pw2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(1e-6 * torch.ones(c))

    def forward(self, x):
        y = self.dw(x).transpose(1, 2)
        y = self.pw2(F.gelu(self.pw1(self.ln(y)))) * self.gamma
        return x + y.transpose(1, 2)


class ConvNeXt(nn.Module):
    def __init__(self, c_in, d=128, dims=(64, 128, 192), depths=(2, 2, 2)):
        super().__init__()
        layers = [nn.Conv1d(c_in, dims[0], 4, 4), ChannelLN(dims[0])]       # patchify stem
        for i, (w, n) in enumerate(zip(dims, depths)):
            if i > 0:
                layers += [ChannelLN(dims[i - 1]), nn.Conv1d(dims[i - 1], w, 2, 2)]
            layers += [ConvNeXtBlock(w) for _ in range(n)]
        self.body = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dims[-1])
        self.fc = nn.Linear(dims[-1], d)

    def forward(self, x):
        if x.shape[-1] < 32:                          # very short windows (EMG): upsample
            x = F.interpolate(x, size=64, mode="linear", align_corners=False)
        return self.fc(self.norm(self.body(x).mean(-1)))


# =============================================================== RNN / GRU / LSTM
class AttnPool(nn.Module):
    """Additive (Bahdanau) attention over time steps -> context vector."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))

    def forward(self, h):
        a = torch.softmax(self.score(h), 1)
        return (a * h).sum(1)


class Recurrent(nn.Module):
    """Many-to-one recurrent encoder. Sequence is average-pooled to <= 64 steps first."""

    def __init__(self, c_in, d=128, cell="lstm", bidirectional=True, hidden=96, layers=2,
                 p=0.2, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.inp = nn.Linear(c_in, hidden)
        rnn = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional, dropout=p)
        h = hidden * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(h)
        self.pool = AttnPool(h)
        self.fc = nn.Linear(h, d)

    def forward(self, x):
        if x.shape[-1] > self.max_len:
            x = adaptive_avg_pool1d(x, self.max_len)
        h, _ = self.rnn(self.inp(x.transpose(1, 2)))
        return self.fc(self.pool(self.norm(h)))


# =============================================================== Transformer / ViT / BERT
def sinusoid(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def tf_encoder(d, depth, heads=4, p=0.1):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, p, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class Transformer(nn.Module):
    """Vanilla Transformer encoder: one token per (pooled) time step, sinusoidal positions,
    mean pooling over tokens."""

    def __init__(self, c_in, d=128, n_tokens=64, depth=4):
        super().__init__()
        self.n = n_tokens
        self.proj = nn.Linear(c_in, d)
        self.register_buffer("pe", sinusoid(n_tokens, d))
        self.enc = tf_encoder(d, depth)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        t = adaptive_avg_pool1d(x, self.n).transpose(1, 2)
        return self.norm(self.enc(self.proj(t) + self.pe).mean(1))


class ViT(nn.Module):
    """Vision Transformer on 1-D signals: window resampled to n_patches*patch samples, split into
    non-overlapping patches, linear patch embedding, [CLS] token, learned positions."""

    def __init__(self, c_in, d=128, n_patches=32, patch=8, depth=4):
        super().__init__()
        self.n, self.p = n_patches, patch
        self.embed = nn.Linear(c_in * patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d) * 0.02)
        self.enc = tf_encoder(d, depth)
        self.norm = nn.LayerNorm(d)

    def patches(self, x):
        x = F.interpolate(x, size=self.n * self.p, mode="linear", align_corners=False)
        B, C, _ = x.shape
        return x.view(B, C, self.n, self.p).permute(0, 2, 1, 3).reshape(B, self.n, C * self.p)

    def tokens(self, x):
        return self.embed(self.patches(x))

    def encode(self, tok):
        t = torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1) + self.pos
        return self.norm(self.enc(t))

    def forward(self, x):
        return self.encode(self.tokens(x))[:, 0]


class BERT(ViT):
    """BERT-style encoder ([CLS] + bidirectional self-attention). Same backbone as ViT but
    pre-trained with masked-token modelling: random patches are replaced by a learned [MASK]
    embedding and the model reconstructs them."""

    def __init__(self, c_in, d=128, n_patches=32, patch=8, depth=4):
        super().__init__(c_in, d, n_patches, patch, depth)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.mlm_head = nn.Linear(d, c_in * patch)

    def mlm_loss(self, x, ratio=0.15):
        target = self.patches(x)
        tok = self.embed(target)
        m = torch.rand(tok.shape[:2], device=x.device) < ratio
        if not m.any():
            m[0, 0] = True
        tok = torch.where(m.unsqueeze(-1), self.mask_tok.expand_as(tok), tok)
        rec = self.mlm_head(self.encode(tok)[:, 1:])
        return F.mse_loss(rec[m], target[m])


# =============================================================== Conv-AE / AE+LSTM / MT-Transformer encoders
class ConvAEEncoder(nn.Module):
    """1-D Convolutional Autoencoder backbone: strided Conv1d stack preserving temporal feature map."""

    def __init__(self, c_in, d=128):
        super().__init__()
        self.conv = nn.Sequential(
            cbr(c_in, 64, 5, 2),
            cbr(64, 96, 5, 2),
            cbr(96, d, 3, 2),
        )
        self.head = Head(d, d)

    def conv_map(self, x):
        return self.conv(x)

    def forward(self, x):
        return self.head(self.conv(x))


class AELSTMEncoder(nn.Module):
    """AE + LSTM encoder: 1-D Conv Autoencoder compresses local time steps into a latent sequence
    (while reconstructing the signal via an auxiliary decoder), followed by a 2-layer LSTM."""

    def __init__(self, c_in, d=128, latent_ch=64, hidden=96):
        super().__init__()
        self.ae_enc = nn.Sequential(
            cbr(c_in, 64, 5, 2),
            cbr(64, latent_ch, 5, 2),
        )
        self.ae_dec = nn.Sequential(
            nn.ConvTranspose1d(latent_ch, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, c_in, 4, 2, 1),
        )
        self.lstm = nn.LSTM(latent_ch, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.norm = nn.LayerNorm(2 * hidden)
        self.pool = AttnPool(2 * hidden)
        self.fc = nn.Linear(2 * hidden, d)

    def ae_loss(self, x):
        z = self.ae_enc(x)
        rec = F.interpolate(self.ae_dec(z), size=x.shape[-1], mode="linear", align_corners=False)
        return F.mse_loss(rec, x)

    def seq_features(self, x):
        z = self.ae_enc(x)                                   # (B, latent_ch, T')
        h, _ = self.lstm(z.transpose(1, 2))                  # (B, T', 2*hidden)
        return self.norm(h), z

    def forward(self, x):
        h, _ = self.seq_features(x)
        return self.fc(self.pool(h))


class MultiTaskTransformerEncoder(nn.Module):
    """Multi-task Transformer encoder with dedicated task tokens ([CLS_SPEED] and [CLS_LOAD])
    over 1-D patch embeddings."""

    def __init__(self, c_in, d=128, n_patches=32, patch=8, depth=4):
        super().__init__()
        self.n, self.p = n_patches, patch
        self.embed = nn.Linear(c_in * patch, d)
        self.cls_speed = nn.Parameter(torch.zeros(1, 1, d))
        self.cls_load = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 2, d) * 0.02)
        self.enc = tf_encoder(d, depth)
        self.norm = nn.LayerNorm(d)

    def encode_tokens(self, x):
        x = F.interpolate(x, size=self.n * self.p, mode="linear", align_corners=False)
        B, C, _ = x.shape
        patches = x.view(B, C, self.n, self.p).permute(0, 2, 1, 3).reshape(B, self.n, C * self.p)
        tok = self.embed(patches)
        t = torch.cat([self.cls_speed.expand(B, -1, -1),
                       self.cls_load.expand(B, -1, -1), tok], 1) + self.pos
        return self.norm(self.enc(t))

    def forward(self, x):
        h = self.encode_tokens(x)
        return 0.5 * (h[:, 0] + h[:, 1])


ENCODERS = {
    "mlp": MLP,
    "mlp_wn": lambda c, d: MLP(c, d, norm="wn"),
    "vgg": VGG, "inception": Inception, "resnet": ResNet,
    "resnet_dilated": lambda c, d: ResNet(c, d, dilated=True), "densenet": DenseNet,
    "mobilenet": MobileNet, "efficientnet": EfficientNet, "convnext": ConvNeXt,
    "rnn": lambda c, d: Recurrent(c, d, "rnn"),
    "gru": lambda c, d: Recurrent(c, d, "gru"),
    "lstm": lambda c, d: Recurrent(c, d, "lstm"),
    "lstm_uni": lambda c, d: Recurrent(c, d, "lstm", bidirectional=False),
    "transformer": Transformer, "vit": ViT, "bert": BERT,
    "conv_ae": ConvAEEncoder,
    "ae_lstm": AELSTMEncoder,
    "mt_transformer": MultiTaskTransformerEncoder,
}


# =============================================================== Fusion classifier
class FusionNet(nn.Module):
    """One encoder per modality -> modality tokens -> Transformer encoder (self-attention across
    modalities) -> MLP classifier. Modality dropout regularises the fusion."""

    def __init__(self, in_ch: dict, encoder="resnet", d=128, n_classes=3, p=0.3, mod_drop=0.2):
        super().__init__()
        self.mods = list(in_ch)
        self.enc = nn.ModuleDict({m: ENCODERS[encoder](c, d) for m, c in in_ch.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(self.mods), d) * 0.02)
        self.mod_drop = mod_drop
        if len(self.mods) > 1:
            self.fuse = tf_encoder(d, 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(d, n_classes))
        self.apply(init_weights)

    def encode_fused(self, batch: dict):
        toks = torch.stack([self.enc[m](batch[m]) for m in self.mods], 1)
        if len(self.mods) == 1:
            return toks[:, 0]
        toks = toks + self.mod_emb
        if self.training and self.mod_drop > 0:
            keep = (torch.rand(toks.shape[:2], device=toks.device) > self.mod_drop).float()
            keep[keep.sum(1) == 0, 0] = 1.0
            toks = toks * keep.unsqueeze(-1)
        return self.fuse(toks).mean(1)

    def forward(self, batch: dict):
        return self.head(self.encode_fused(batch))


# =============================================================== Autoencoders / Conv-AE / VAE
class AutoEncoder(nn.Module):
    """Unsupervised pre-training and anomaly-detection wrapper around any encoder.

    kind = "ae"      : basic autoencoder (reconstruct the input)
           "dae"     : denoising autoencoder (Gaussian noise + time masking on the input)
           "sae"     : sparse autoencoder (L1 sparsity penalty on the latent code)
           "conv_ae" : 1-D convolutional autoencoder (strided conv bottleneck -> transposed conv decoder)
           "vae"     : variational autoencoder (mu/log-var heads, reparameterisation, KL term)
    """

    def __init__(self, encoder: nn.Module, c_in, length, kind="dae", d=128):
        super().__init__()
        self.encoder, self.kind, self.L = encoder, kind, length
        self.t0 = max(1, math.ceil(length / 8))
        if kind == "vae":
            self.mu, self.logvar = nn.Linear(d, d), nn.Linear(d, d)
        if kind == "conv_ae":
            self.conv_enc = (encoder.conv if isinstance(encoder, ConvAEEncoder) else
                             nn.Sequential(cbr(c_in, 64, 5, 2), cbr(64, 96, 5, 2), cbr(96, d, 3, 2)))
            self.conv_proj = nn.Conv1d(d, 64, 1)
        self.fc = nn.Linear(d, 64 * self.t0)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, c_in, 3, 1, 1))
        extra = [self.fc, self.dec]
        if kind == "vae":
            extra += [self.mu, self.logvar]
        if kind == "conv_ae":
            extra += [self.conv_proj]
        for m in extra:
            m.apply(init_weights)

    def decode(self, z):
        r = self.dec(self.fc(z).view(len(z), 64, self.t0))
        return F.interpolate(r, size=self.L, mode="linear", align_corners=False)

    def decode_conv(self, fmap):
        r = self.dec(self.conv_proj(fmap))
        return F.interpolate(r, size=self.L, mode="linear", align_corners=False)

    def reconstruct(self, x):
        if self.kind == "conv_ae":
            fmap = self.conv_enc(x)
            # Also exercise encoder forward so arbitrary encoder weights train when wrapped
            h = self.encoder(x)
            rec = self.decode_conv(fmap) + 0.1 * self.decode(h)
            return rec, fmap
        h = self.encoder(x)
        if self.kind == "vae":
            mu = self.mu(h)
            return self.decode(mu), mu
        return self.decode(h), h

    @torch.no_grad()
    def recon_errors(self, x):
        """Return per-window (mse, mae, anomaly_score) tensors of shape (B,)."""
        rec, h = self.reconstruct(x)
        diff = rec - x
        mse = diff.pow(2).mean(dim=(1, 2))
        mae = diff.abs().mean(dim=(1, 2))
        score = mse + (1e-3 * h.abs().mean(dim=tuple(range(1, h.ndim))) if self.kind == "sae" else 0.0)
        return mse, mae, score

    def forward(self, x):
        xin = x
        if self.kind == "dae":
            xin = x + 0.3 * torch.randn_like(x)
            xin = xin * (torch.rand_like(x[:, :1]) > 0.15).float()
        if self.kind == "conv_ae":
            rec, fmap = self.reconstruct(xin)
            return F.mse_loss(rec, x) + 0.1 * F.l1_loss(rec, x)
        h = self.encoder(xin)
        if self.kind == "vae":
            mu, lv = self.mu(h), self.logvar(h).clamp(-8, 8)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            return F.mse_loss(self.decode(z), x) + 1e-2 * kl
        loss = F.mse_loss(self.decode(h), x)
        if self.kind == "sae":
            loss = loss + 5e-3 * h.abs().mean()
        return loss


# =============================================================== BiGAN (Bidirectional GAN)
SN = nn.utils.parametrizations.spectral_norm


class BiGAN(nn.Module):
    """Bidirectional GAN (Donahue et al., 2017) for unsupervised representation learning &
    anomaly detection.

    Jointly trains:
      - Encoder E(x) : maps real window x -> latent code z_hat in R^d
      - Generator G(z): maps prior sample z ~ N(0, I) -> synthetic window x_hat
      - Joint Discriminator D(x, z): discriminates (x, E(x)) vs (G(z), z) with spectral norm.
    """

    def __init__(self, encoder: nn.Module, c_in, length, d=128):
        super().__init__()
        self.E = encoder
        self.L, self.d = length, d
        self.t0 = max(1, math.ceil(length / 8))
        # Generator G(z) -> (B, c_in, L)
        self.g_fc = nn.Linear(d, 64 * self.t0)
        self.G = nn.Sequential(
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, c_in, 3, 1, 1),
        )
        # Joint Discriminator D(x, z)
        self.d_x = nn.Sequential(
            SN(nn.Conv1d(c_in, 64, 5, 2, 2)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(64, 128, 5, 2, 2)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(128, 128, 5, 2, 2)), nn.LeakyReLU(0.2),
        )
        self.d_z = nn.Sequential(
            SN(nn.Linear(d, 128)), nn.LeakyReLU(0.2),
            SN(nn.Linear(128, 128)), nn.LeakyReLU(0.2),
        )
        self.d_joint = nn.Sequential(
            SN(nn.Linear(256, 128)), nn.LeakyReLU(0.2),
        )
        self.d_out = SN(nn.Linear(128, 1))
        for m in (self.g_fc, self.G):
            m.apply(init_weights)

    def generate(self, z):
        h = self.g_fc(z).view(len(z), 64, self.t0)
        return F.interpolate(self.G(h), size=self.L, mode="linear", align_corners=False)

    def disc_features(self, x, z):
        fx = self.d_x(x).mean(-1)
        fz = self.d_z(z)
        return self.d_joint(torch.cat([fx, fz], 1))

    def discriminate(self, x, z):
        feat = self.disc_features(x, z)
        return self.d_out(feat).squeeze(1), feat

    def reconstruct(self, x):
        z_hat = self.E(x)
        return self.generate(z_hat), z_hat

    @torch.no_grad()
    def recon_errors(self, x, alpha=0.2):
        """Per-sample (mse, mae, bigan_anomaly_score) combining reconstruction & D feature distance."""
        x_rec, z_hat = self.reconstruct(x)
        diff = x_rec - x
        mse = diff.pow(2).mean(dim=(1, 2))
        mae = diff.abs().mean(dim=(1, 2))
        _, f_real = self.discriminate(x, z_hat)
        _, f_rec = self.discriminate(x_rec, z_hat)
        feat_err = (f_real - f_rec).abs().mean(dim=1)
        score = (1.0 - alpha) * mae + alpha * feat_err
        return mse, mae, score


# =============================================================== Conditional GAN (Class-Conditional)
class CondGenerator(nn.Module):
    """Class-conditional generator: (noise z, speed label) -> one synthetic window per modality.
    Shared transposed-conv trunk at a 64-step time base, one output head per modality that is
    resampled to that modality's window length."""

    def __init__(self, shapes: dict, n_classes=3, z_dim=64, h=128):
        super().__init__()
        self.shapes, self.z_dim = shapes, z_dim
        self.emb = nn.Embedding(n_classes, z_dim)
        self.fc = nn.Linear(2 * z_dim, h * 16)
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(h), nn.ReLU(),
            nn.ConvTranspose1d(h, h, 4, 2, 1), nn.BatchNorm1d(h), nn.ReLU(),   # 16 -> 32
            nn.ConvTranspose1d(h, h, 4, 2, 1), nn.BatchNorm1d(h), nn.ReLU(),   # 32 -> 64
            nn.Conv1d(h, h, 5, 1, 2), nn.BatchNorm1d(h), nn.ReLU())
        self.heads = nn.ModuleDict({m: nn.Sequential(nn.Conv1d(h, h, 5, 1, 2), nn.ReLU(),
                                                     nn.Conv1d(h, c, 5, 1, 2))
                                    for m, (c, _) in shapes.items()})
        self.h = h

    def forward(self, y):
        z = torch.randn(len(y), self.z_dim, device=y.device)
        t = self.trunk(self.fc(torch.cat([z, self.emb(y)], 1)).view(len(y), self.h, 16))
        out = {}
        for m, (_, L) in self.shapes.items():
            x = F.interpolate(t, size=L, mode="linear", align_corners=False)
            out[m] = self.heads[m](x)
        return out


class CondDiscriminator(nn.Module):
    """Projection discriminator with spectral normalisation (stabilises GAN training).
    One strided-conv branch per modality -> concatenated features -> real/fake score,
    conditioned on the label via an inner product with a class embedding."""

    def __init__(self, shapes: dict, n_classes=3, h=64):
        super().__init__()
        self.branch = nn.ModuleDict({m: nn.Sequential(
            SN(nn.Conv1d(c, h, 5, 2, 2)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(h, 2 * h, 5, 2, 2)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(2 * h, 2 * h, 5, 2, 2)), nn.LeakyReLU(0.2)) for m, (c, _) in shapes.items()})
        f = 2 * h * len(shapes)
        self.out = SN(nn.Linear(f, 1))
        self.emb = SN(nn.Embedding(n_classes, f))

    def forward(self, x: dict, y):
        feat = torch.cat([self.branch[m](x[m]).mean(-1) for m in self.branch], 1)
        return self.out(feat).squeeze(1) + (self.emb(y) * feat).sum(1)


# =============================================================== Probabilistic Load & Multi-Task Forecasters
class AELSTMForecaster(nn.Module):
    """AE + LSTM multi-step biomechanical load forecaster & speed classifier.
    1. Per-modality Conv Autoencoder compresses context signals into latent sequences & reconstructs input.
    2. Cross-modal latent sequence is processed by a 2-layer LSTM.
    3. Outputs predictive mean mu(t), log-std log_sigma(t) over horizon H, and speed class logits.
    """

    def __init__(self, in_ch: dict, horizon=32, d=128, n_classes=3):
        super().__init__()
        self.mods = list(in_ch)
        self.horizon = horizon
        self.enc = nn.ModuleDict({m: AELSTMEncoder(c, d) for m, c in in_ch.items()})
        self.fuse_proj = nn.Linear(len(self.mods) * d, d)
        self.dec_lstm = nn.LSTM(d, d, num_layers=2, batch_first=True, dropout=0.2)
        self.q_pos = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.mu_head = nn.Linear(d, 1)
        self.logstd_head = nn.Linear(d, 1)
        self.cls_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
        self.apply(init_weights)

    def ae_recon_loss(self, batch: dict):
        return sum(self.enc[m].ae_loss(batch[m]) for m in self.mods) / len(self.mods)

    def forward(self, batch: dict):
        embs = [self.enc[m](batch[m]) for m in self.mods]
        ctx = F.gelu(self.fuse_proj(torch.cat(embs, 1)))           # (B, d)
        logits = self.cls_head(ctx)
        q = ctx.unsqueeze(1).expand(-1, self.horizon, -1) + self.q_pos
        h, _ = self.dec_lstm(q)
        mu = self.mu_head(h).squeeze(-1)                           # (B, H)
        sigma = F.softplus(self.logstd_head(h).squeeze(-1)) + 1e-3
        return {"logits": logits, "mu": mu, "sigma": sigma}


class MultiTaskTransformerForecaster(nn.Module):
    """Multi-task Transformer jointly predicting:
      - Task 1: Walking speed classification logits (0.5 / 0.75 / 1.0 m/s)
      - Task 2: Future biomechanical load trajectory distribution (mu, sigma) over horizon H
      - Task 3: Peak load & ramp-intensity auxiliary predictions
    """

    def __init__(self, in_ch: dict, horizon=32, d=128, n_classes=3):
        super().__init__()
        self.mods = list(in_ch)
        self.horizon = horizon
        self.enc = nn.ModuleDict({m: MultiTaskTransformerEncoder(c, d) for m, c in in_ch.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(self.mods), 1, d) * 0.02)
        self.horizon_queries = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.shared_tf = tf_encoder(d, depth=2, heads=4)
        self.cls_head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, n_classes))
        self.mu_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.logstd_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.peak_ramp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.apply(init_weights)

    def forward(self, batch: dict):
        tok_list = []
        for i, m in enumerate(self.mods):
            tok_list.append(self.enc[m].encode_tokens(batch[m]) + self.mod_emb[i])
        ctx_tokens = torch.cat(tok_list, dim=1)                    # (B, M*(N+2), d)
        B = ctx_tokens.shape[0]
        q = self.horizon_queries.expand(B, -1, -1)
        all_tok = self.shared_tf(torch.cat([ctx_tokens, q], dim=1))
        speed_tok = all_tok[:, 0]
        load_tok = all_tok[:, 1]
        h_tok = all_tok[:, -self.horizon:]
        logits = self.cls_head(speed_tok)
        mu = self.mu_head(h_tok).squeeze(-1)
        sigma = F.softplus(self.logstd_head(h_tok).squeeze(-1)) + 1e-3
        pr = self.peak_ramp_head(load_tok)
        return {"logits": logits, "mu": mu, "sigma": sigma, "peak_pred": pr[:, 0], "ramp_pred": pr[:, 1]}


class CondGANForecaster(nn.Module):
    """Conditional GAN (cGAN) probabilistic load forecaster.
      - Generator G(x_ctx, z): conditioned on multimodal context embedding, maps latent noise z
        to realistic future load trajectories y_hat(1..H).
      - Projection Discriminator D(x_ctx, y): spectral-norm 1-D Conv discriminator scoring whether
        a load curve is real or synthetic given context x_ctx.
      - At evaluation, draws M Monte Carlo trajectories to form an empirical distribution for CRPS.
    """

    def __init__(self, in_ch: dict, horizon=32, d=128, z_dim=32, n_classes=3):
        super().__init__()
        self.mods = list(in_ch)
        self.horizon, self.z_dim, self.d = horizon, z_dim, d
        self.ctx_enc = FusionNet(in_ch, encoder="resnet", d=d, n_classes=n_classes, mod_drop=0.0)
        # Generator: (ctx, z) -> (B, H)
        self.g_fc = nn.Linear(d + z_dim, 64 * (horizon // 4))
        self.g_net = nn.Sequential(
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 3, 1, 1),
        )
        # Conditional Discriminator D(ctx, y_seq)
        self.d_seq = nn.Sequential(
            SN(nn.Conv1d(1, 32, 5, 2, 2)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(32, 64, 5, 2, 2)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(64, d, 3, 1, 1)), nn.LeakyReLU(0.2),
        )
        self.d_out = SN(nn.Linear(d, 1))
        self.d_proj = SN(nn.Linear(d, d))

    def encode_context(self, batch: dict):
        ctx = self.ctx_enc.encode_fused(batch)
        logits = self.ctx_enc.head(ctx)
        return ctx, logits

    def sample_trajectories(self, ctx, n_samples=1):
        B = ctx.shape[0]
        ctx_rep = ctx.unsqueeze(1).expand(B, n_samples, self.d).reshape(B * n_samples, self.d)
        z = torch.randn(B * n_samples, self.z_dim, device=ctx.device)
        h = self.g_fc(torch.cat([ctx_rep, z], 1)).view(B * n_samples, 64, self.horizon // 4)
        y = F.interpolate(self.g_net(h), size=self.horizon, mode="linear", align_corners=False)
        return y.view(B, n_samples, self.horizon)

    def discriminate(self, ctx, y_seq):
        feat = self.d_seq(y_seq.unsqueeze(1)).mean(-1)             # (B, d)
        return self.d_out(feat).squeeze(1) + (self.d_proj(ctx) * feat).sum(1)

    def forward(self, batch: dict, n_samples=24):
        ctx, logits = self.encode_context(batch)
        samples = self.sample_trajectories(ctx, n_samples=n_samples)
        mu = samples.mean(dim=1)
        sigma = samples.std(dim=1) + 1e-3
        return {"logits": logits, "mu": mu, "sigma": sigma, "samples": samples}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    shapes = {"IMU": (204, 240), "FP": (16, 400), "EMG": (12, 64)}
    dummy = {k: torch.randn(4, *v) for k, v in shapes.items()}
    for name in ENCODERS:
        net = FusionNet({k: v[0] for k, v in shapes.items()}, encoder=name)
        out = net(dummy)
        print(f"{name:15s} out={tuple(out.shape)} params={count_params(net) / 1e6:.2f}M")
    for kind in ("ae", "dae", "sae", "conv_ae", "vae"):
        ae = AutoEncoder(ResNet(204), 204, 240, kind)
        mse, mae, sc = ae.recon_errors(torch.randn(4, 204, 240))
        print(kind, float(ae(torch.randn(4, 204, 240)).detach()), float(mse.mean()), float(mae.mean()))
    bg = BiGAN(ResNet(204), 204, 240)
    mse, mae, sc = bg.recon_errors(torch.randn(4, 204, 240))
    print("bigan", float(mse.mean()), float(mae.mean()), float(sc.mean()))
    for cls in (AELSTMForecaster, MultiTaskTransformerForecaster, CondGANForecaster):
        fnet = cls({k: v[0] for k, v in shapes.items()})
        fout = fnet(dummy)
        print(cls.__name__, tuple(fout["mu"].shape), tuple(fout["logits"].shape))
    print("bert mlm", float(BERT(16).mlm_loss(torch.randn(4, 16, 400)).detach()))
