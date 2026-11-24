"""Syllabus model zoo (1-D adaptations for multichannel time series).

Every encoder maps one modality window (B, C, T) to an embedding (B, d).
Only architectures named in the course syllabus are used:

  Deep feed-forward network ............ MLP
  CNN architectures .................... VGG, Inception, ResNet, DenseNet, MobileNet,
                                         EfficientNet, ConvNeXt
  Recurrent models ..................... RNN, GRU, LSTM (bidirectional, many-to-one,
                                         attention pooling) + unidirectional LSTM
  Attention models ..................... Transformer encoder, Vision Transformer (ViT), BERT
  Deep generative (pre-training) ....... Autoencoder (basic / denoising / sparse), VAE,
                                         BERT masked-token modelling

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
        self.pool = nn.AdaptiveAvgPool1d(steps)
        wn = nn.utils.parametrizations.weight_norm

        def fc(i, o):
            if norm == "wn":
                return [wn(nn.Linear(i, o)), nn.ReLU(), nn.Dropout(p)]
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p)]
        self.net = nn.Sequential(nn.Flatten(), *fc(c_in * steps, 512), *fc(512, 256),
                                 nn.Linear(256, d))

    def forward(self, x):
        return self.net(self.pool(x))


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
            x = F.adaptive_avg_pool1d(x, self.max_len)
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
        t = F.adaptive_avg_pool1d(x, self.n).transpose(1, 2)
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
        tok = torch.where(m.unsqueeze(-1), self.mask_tok.expand_as(tok), tok)
        rec = self.mlm_head(self.encode(tok)[:, 1:])
        return F.mse_loss(rec[m], target[m])


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

    def forward(self, batch: dict):
        toks = torch.stack([self.enc[m](batch[m]) for m in self.mods], 1)
        if len(self.mods) == 1:
            return self.head(toks[:, 0])
        toks = toks + self.mod_emb
        if self.training and self.mod_drop > 0:
            keep = (torch.rand(toks.shape[:2], device=toks.device) > self.mod_drop).float()
            keep[keep.sum(1) == 0, 0] = 1.0
            toks = toks * keep.unsqueeze(-1)
        return self.head(self.fuse(toks).mean(1))


# =============================================================== Autoencoders / VAE
class AutoEncoder(nn.Module):
    """Unsupervised pre-training wrapper around any encoder.

    kind = "ae"  : basic autoencoder (reconstruct the input)
           "dae" : denoising autoencoder (Gaussian noise + time masking on the input)
           "sae" : sparse autoencoder (L1 penalty on the latent code)
           "vae" : variational autoencoder (mu/log-var heads, reparameterisation, KL term)
    """

    def __init__(self, encoder: nn.Module, c_in, length, kind="dae", d=128):
        super().__init__()
        self.encoder, self.kind, self.L = encoder, kind, length
        self.t0 = max(1, math.ceil(length / 8))
        if kind == "vae":
            self.mu, self.logvar = nn.Linear(d, d), nn.Linear(d, d)
        self.fc = nn.Linear(d, 64 * self.t0)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, c_in, 3, 1, 1))
        for m in [self.fc, self.dec] + ([self.mu, self.logvar] if kind == "vae" else []):
            m.apply(init_weights)

    def decode(self, z):
        r = self.dec(self.fc(z).view(len(z), 64, self.t0))
        return F.interpolate(r, size=self.L, mode="linear", align_corners=False)

    def forward(self, x):
        xin = x
        if self.kind == "dae":
            xin = x + 0.3 * torch.randn_like(x)
            xin = xin * (torch.rand_like(x[:, :1]) > 0.15).float()
        h = self.encoder(xin)
        if self.kind == "vae":
            mu, lv = self.mu(h), self.logvar(h).clamp(-8, 8)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            return F.mse_loss(self.decode(z), x) + 1e-2 * kl
        loss = F.mse_loss(self.decode(h), x)
        if self.kind == "sae":
            loss = loss + 1e-3 * h.abs().mean()
        return loss


# =============================================================== Conditional GAN
SN = nn.utils.parametrizations.spectral_norm


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


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    shapes = {"IMU": (204, 240), "FP": (16, 400), "EMG": (12, 64)}
    for name in ENCODERS:
        net = FusionNet({k: v[0] for k, v in shapes.items()}, encoder=name)
        out = net({k: torch.randn(4, *v) for k, v in shapes.items()})
        print(f"{name:13s} out={tuple(out.shape)} params={count_params(net) / 1e6:.2f}M")
    for kind in ("ae", "dae", "sae", "vae"):
        ae = AutoEncoder(ResNet(204), 204, 240, kind)
        print(kind, float(ae(torch.randn(4, 204, 240)).detach()))
    print("bert mlm", float(BERT(16).mlm_loss(torch.randn(4, 16, 400)).detach()))
