"""Subject-independent walking-speed classification (0.5 / 0.75 / 1.0 m/s).

Evaluation protocol: K-fold cross-validation grouped by SUBJECT, so no person in a test fold
was seen during training. Inside each training fold ~15% of subjects are held out for early
stopping / model selection. Metrics are reported per window and per trial (averaging the
softmax over every window of a 60-s recording).

Example:
  python src/train.py --mods IMU FP EMG --encoder resnet --epochs 40
"""
import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import erf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from models import (ENCODERS, AELSTMForecaster, AutoEncoder, BiGAN, CondDiscriminator,
                    CondGANForecaster, CondGenerator, FusionNet, MultiTaskTransformerForecaster,
                    count_params)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FS = {"EMG": 16, "FP": 100, "IMU": 60, "EEG": 100}
SPEEDS = ["0.5", "0.75", "1"]


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEV = default_device()


def amp_ctx():
    if DEV.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def empty_device_cache():
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    elif DEV.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------- data
def load_trials(mods):
    missing = [m for m in mods if not (PROC / m).exists() or not any((PROC / m).glob("S*.npy"))]
    if missing:
        print(f"Preprocessed data missing for {missing}; running prepare_data.py automatically...")
        import prepare_data
        prepare_data.run(missing)
    keys = None
    for m in mods:
        ks = {p.stem for p in (PROC / m).glob("S*.npy")}
        keys = ks if keys is None else keys & ks
    if not keys:
        raise RuntimeError(
            f"No preprocessed .npy trials found in {PROC} for modalities {mods}. "
            "Check that raw dataset files exist in data/ or data/Dataset/."
        )
    keys = sorted(keys, key=lambda k: (int(k.split("_")[0][1:]), float(k.split("_")[1])))
    data = {m: [np.load(PROC / m / f"{k}.npy") for k in keys] for m in mods}
    subj = np.array([int(k.split("_")[0][1:]) for k in keys])
    y = np.array([SPEEDS.index(k.split("_")[1]) for k in keys])
    dur = np.array([min(len(data[m][i]) / FS[m] for m in mods) for i in range(len(keys))])
    return keys, data, subj, y, dur


def subject_folds(subjects, k, seed):
    rng = np.random.RandomState(seed)
    u = np.unique(subjects)
    rng.shuffle(u)
    return [u[i::k] for i in range(k)]


class WindowBank:
    """All trials of one modality on the GPU, padded to (N, C, Tmax); windows are gathered by index."""

    def __init__(self, arrays, mean, std, fs, win):
        self.fs, self.L = fs, int(round(win * fs))
        tmax = max(len(a) for a in arrays)
        x = np.zeros((len(arrays), arrays[0].shape[1], tmax), np.float32)
        for i, a in enumerate(arrays):
            x[i, :, : len(a)] = ((a - mean) / std).T
        self.x = torch.from_numpy(x).to(DEV)
        self.ar = torch.arange(self.L, device=DEV)

    def get(self, trial_idx, start_sec):
        s = torch.round(start_sec * self.fs).long()
        idx = (s[:, None] + self.ar[None]).clamp_max(self.x.shape[-1] - 1)      # (B, L)
        xb = self.x[trial_idx]                                                   # (B, C, T)
        return torch.gather(xb, 2, idx[:, None, :].expand(-1, xb.shape[1], -1))


def make_windows(trials, dur, win, hop):
    tr, st = [], []
    for i in trials:
        n = int((dur[i] - win) // hop) + 1
        tr += [i] * n
        st += list(np.arange(n) * hop)
    return np.array(tr), np.array(st, np.float32)


def augment(x, noise=0.05, scale=0.1):
    x = x * (1 + scale * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))
    return x + noise * torch.randn_like(x)


# ----------------------------------------------------------------------------- training
def make_optimizer(name, params, lr, wd):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def get_batch(banks, tr, st, sel, train, hop):
    t = torch.from_numpy(tr[sel]).to(DEV)
    s = torch.from_numpy(st[sel]).to(DEV)
    if train:   # random temporal crop: jitter the window start inside its hop
        s = s + torch.rand_like(s) * hop
    out = {}
    for m, b in banks.items():
        x = b.get(t, s)
        out[m] = augment(x) if train else x
    return out


@torch.no_grad()
def predict(model, banks, tr, st, hop, bs=512):
    model.eval()
    probs = []
    for sel in batches(len(tr), bs, False, None):
        with amp_ctx():
            logits = model(get_batch(banks, tr, st, sel, False, hop))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(probs)


def safe_auc_roc(y_true, probs):
    """Multi-class macro AUC-ROC (one-vs-rest) or binary AUC-ROC."""
    try:
        if probs.ndim == 1 or probs.shape[1] == 1:
            return float(roc_auc_score(y_true, probs))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def best_binary_f1(y_true, scores):
    """Compute binary F1 across candidate percentile thresholds on anomaly scores."""
    best = 0.0
    for q in np.linspace(50, 90, 41):
        thr = np.percentile(scores, q)
        f1 = f1_score(y_true, (scores >= thr).astype(int), zero_division=0)
        if f1 > best:
            best = float(f1)
    return best


def train_bigan_modality(enc, b, tr, st, args, rng):
    """Train a Bidirectional GAN (BiGAN) for one modality on training windows."""
    bigan = BiGAN(enc, b.x.shape[1], b.L, args.d).to(DEV)
    eg_params = list(bigan.E.parameters()) + list(bigan.g_fc.parameters()) + list(bigan.G.parameters())
    d_params = (list(bigan.d_x.parameters()) + list(bigan.d_z.parameters()) +
                list(bigan.d_joint.parameters()) + list(bigan.d_out.parameters()))
    oEG = torch.optim.AdamW(eg_params, lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    oD = torch.optim.AdamW(d_params, lr=2e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    for ep in range(args.pre_epochs):
        bigan.train()
        tot_rec = 0.0
        for sel in batches(len(tr), args.bs, True, rng):
            x = get_batch({"m": b}, tr, st, sel, True, args.hop)["m"]
            z = torch.randn(len(x), args.d, device=DEV)
            z_hat = bigan.E(x)
            x_hat = bigan.generate(z)
            # Discriminator step: (x, E(x)) -> 1, (G(z), z) -> 0
            d_enc, _ = bigan.discriminate(x, z_hat.detach())
            d_gen, _ = bigan.discriminate(x_hat.detach(), z)
            loss_d = (F.binary_cross_entropy_with_logits(d_enc, torch.full_like(d_enc, 0.9)) +
                      F.binary_cross_entropy_with_logits(d_gen, torch.zeros_like(d_gen)))
            oD.zero_grad(set_to_none=True)
            loss_d.backward()
            oD.step()
            # Encoder + Generator step: adversarial fooling + bidirectional cyclic reconstruction
            z_hat = bigan.E(x)
            x_hat = bigan.generate(z)
            d_enc, f_real = bigan.discriminate(x, z_hat)
            d_gen, f_fake = bigan.discriminate(x_hat, z)
            adv_loss = (F.binary_cross_entropy_with_logits(d_enc, torch.zeros_like(d_enc)) +
                        F.binary_cross_entropy_with_logits(d_gen, torch.ones_like(d_gen)))
            x_rec = bigan.generate(z_hat)
            rec_loss = F.mse_loss(x_rec, x) + 0.2 * F.l1_loss(x_rec, x)
            loss_eg = rec_loss + 0.1 * adv_loss
            oEG.zero_grad(set_to_none=True)
            loss_eg.backward()
            nn.utils.clip_grad_norm_(eg_params, 1.0)
            oEG.step()
            tot_rec += rec_loss.item() * len(sel)
    print(f"    BIGAN pre-train recon loss {tot_rec / len(tr):.4f}")
    bigan.eval()
    return bigan


@torch.no_grad()
def eval_reconstruction_and_anomaly(reconstructors, banks, va_w, te_w, y, args):
    """Evaluate Reconstruction Error (MSE, MAE) and Anomaly Detection (AUC-ROC, F1) on test fold.

    Evaluates:
      1. Clean test-window reconstruction MSE and MAE.
      2. Unsupervised anomaly detection (normal windows vs anomalous high-impact/corrupted gait
         windows) via reconstruction error, reporting AUC-ROC and F1.
    """
    te_tr, te_st = te_w
    va_tr, va_st = va_w
    mses, maes, norm_scores, anom_scores = [], [], [], []
    rng = np.random.RandomState(args.seed + 999)
    for sel in batches(len(te_tr), 512, False, None):
        xb = get_batch(banks, te_tr, te_st, sel, False, args.hop)
        # Create paired anomalous windows (impulse/drift/high-load gait perturbation + fast gait)
        xb_anom = {}
        for m, x in xb.items():
            noise = torch.from_numpy(rng.normal(0, 1.25, size=x.shape).astype(np.float32)).to(DEV)
            spike = (torch.from_numpy(rng.uniform(0, 1, size=x.shape).astype(np.float32)).to(DEV) > 0.92).float() * 2.5
            xb_anom[m] = x * 1.35 + noise + spike
        b_mse = b_mae = b_sc_norm = b_sc_anom = 0.0
        for m, obj in reconstructors.items():
            obj.eval()
            mse_n, mae_n, sc_n = obj.recon_errors(xb[m])
            _, _, sc_a = obj.recon_errors(xb_anom[m])
            b_mse = b_mse + mse_n.cpu().numpy()
            b_mae = b_mae + mae_n.cpu().numpy()
            b_sc_norm = b_sc_norm + sc_n.cpu().numpy()
            b_sc_anom = b_sc_anom + sc_a.cpu().numpy()
        n_mod = len(reconstructors)
        mses.append(b_mse / n_mod)
        maes.append(b_mae / n_mod)
        norm_scores.append(b_sc_norm / n_mod)
        anom_scores.append(b_sc_anom / n_mod)

    mse_arr = np.concatenate(mses)
    mae_arr = np.concatenate(maes)
    s_norm = np.concatenate(norm_scores)
    s_anom = np.concatenate(anom_scores)

    # Validation threshold selection on normal vs anomalous validation windows
    va_norm, va_anom = [], []
    for sel in batches(len(va_tr), 512, False, None):
        xb = get_batch(banks, va_tr, va_st, sel, False, args.hop)
        b_sn = b_sa = 0.0
        for m, obj in reconstructors.items():
            _, _, sn = obj.recon_errors(xb[m])
            _, _, sa = obj.recon_errors(xb[m] * 1.35 + 1.25 * torch.randn_like(xb[m]))
            b_sn = b_sn + sn.cpu().numpy()
            b_sa = b_sa + sa.cpu().numpy()
        va_norm.append(b_sn / len(reconstructors))
        va_anom.append(b_sa / len(reconstructors))
    va_y = np.concatenate([np.zeros(len(va_tr), dtype=int), np.ones(len(va_tr), dtype=int)])
    va_s = np.concatenate([np.concatenate(va_norm), np.concatenate(va_anom)])
    best_thr, best_va_f1 = np.median(va_s), -1.0
    for q in np.linspace(30, 85, 56):
        thr = np.percentile(va_s, q)
        f1_v = f1_score(va_y, (va_s >= thr).astype(int), zero_division=0)
        if f1_v > best_va_f1:
            best_va_f1, best_thr = f1_v, thr

    y_bin = np.concatenate([np.zeros_like(s_norm, dtype=int), np.ones_like(s_anom, dtype=int)])
    s_all = np.concatenate([s_norm, s_anom])
    auc_anom = safe_auc_roc(y_bin, s_all)
    f1_anom = float(f1_score(y_bin, (s_all >= best_thr).astype(int), zero_division=0))
    return {
        "recon_mse": float(mse_arr.mean()),
        "recon_mae": float(mae_arr.mean()),
        "anomaly_auc_roc": auc_anom,
        "anomaly_f1": f1_anom,
    }


def pretrain(model, banks, tr, st, y, args, rng):
    """Unsupervised pre-training of each modality encoder on the TRAINING windows only
    (labels unused): AE / denoising AE / sparse AE / Conv-AE / VAE / BiGAN reconstruction,
    or BERT masked-token modelling. The pre-trained encoder weights are then fine-tuned with labels."""
    reconstructors = {}
    # If running anomaly task, train reconstruction models primarily on normal/comfortable speeds
    tr_use, st_use = tr, st
    if args.task == "anomaly":
        keep = y[tr] < 2
        if keep.sum() > args.bs:
            tr_use, st_use = tr[keep], st[keep]
    for m, b in banks.items():
        enc = model.enc[m]
        if args.pretrain == "bert":
            obj, loss_fn = enc, enc.mlm_loss
        elif args.pretrain == "bigan":
            reconstructors[m] = train_bigan_modality(enc, b, tr_use, st_use, args, rng)
            continue
        else:
            obj = AutoEncoder(enc, b.x.shape[1], b.L, args.pretrain, args.d).to(DEV)
            loss_fn = obj
            reconstructors[m] = obj
        opt = torch.optim.AdamW(obj.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(args.pre_epochs):
            obj.train()
            tot = 0
            for sel in batches(len(tr_use), args.bs, True, rng):
                x = get_batch({m: b}, tr_use, st_use, sel, True, args.hop)[m]
                with amp_ctx():
                    loss = loss_fn(x)
                opt.zero_grad(set_to_none=True)
                loss.float().backward()
                nn.utils.clip_grad_norm_(obj.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(sel)
        print(f"    {args.pretrain.upper()}[{m}] pre-train loss {tot / len(tr_use):.4f}")
    return reconstructors


def train_gan(banks, tr, st, y, args, rng):
    """Conditional GAN trained on the fold's TRAINING windows only.

    Objective: non-saturating minimax loss (D maximises log D(x) + log(1 - D(G(z))),
    G maximises log D(G(z))). Training-stability measures against the classic GAN
    challenges (instability, mode collapse, vanishing generator gradients):
    spectral normalisation in D, projection conditioning, one-sided label smoothing (0.9),
    Adam(0.5, 0.999) with two-time-scale learning rates (D 4e-4, G 1e-4).
    """
    shapes = {m: (b.x.shape[1], b.L) for m, b in banks.items()}
    G, D = CondGenerator(shapes).to(DEV), CondDiscriminator(shapes).to(DEV)
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 4e-4, betas=(0.5, 0.999))
    yt = torch.from_numpy(y).to(DEV)
    for ep in range(args.gan_epochs):
        G.train(); D.train()
        ld = lg = n = 0
        for sel in batches(len(tr), args.bs, True, rng):
            real = get_batch(banks, tr, st, sel, True, args.hop)
            lab = yt[torch.from_numpy(tr[sel]).to(DEV)]
            fake = G(lab)
            d_real = D(real, lab)
            d_fake = D({k: v.detach() for k, v in fake.items()}, lab)
            loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, 0.9)) +
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            oD.zero_grad(set_to_none=True); loss_d.backward(); oD.step()
            g_out = D(fake, lab)
            loss_g = F.binary_cross_entropy_with_logits(g_out, torch.ones_like(g_out))
            oG.zero_grad(set_to_none=True); loss_g.backward(); oG.step()
            ld += loss_d.item(); lg += loss_g.item(); n += 1
        if ep % 10 == 9 or ep == args.gan_epochs - 1:
            print(f"    GAN epoch {ep + 1}: D loss {ld / n:.3f} | G loss {lg / n:.3f}")
    G.eval()
    return G


# ----------------------------------------------------------------------------- forecasting (AE+LSTM, cGAN, MT-Transformer)
def split_forecast_batch(batch: dict, horizon=32):
    """Split a multimodal window batch into:
      - x_ctx : first 75% of each modality's window (observed history)
      - y_seq : future biomechanical load curve (B, horizon) over the final 25% of the window.
                Uses vertical Ground Reaction Force Fz (left + right belt) if FP is present,
                otherwise the RMS kinematic/muscle load envelope across channels.
    """
    x_ctx = {}
    for m, x in batch.items():
        cut = max(8, int(round(0.75 * x.shape[-1])))
        x_ctx[m] = x[:, :, :cut]
    if "FP" in batch:
        fp = batch["FP"]
        cut = max(8, int(round(0.75 * fp.shape[-1])))
        fut = fp[:, :, cut:]
        # Channels 2 and 10 in FP are 1:Fz and 2:Fz (vertical ground reaction forces)
        c1, c2 = 2 if fut.shape[1] > 2 else 0, 10 if fut.shape[1] > 10 else 0
        raw_load = 0.5 * (fut[:, c1, :] + fut[:, c2, :])
    else:
        first_m = next(iter(batch.values()))
        cut = max(8, int(round(0.75 * first_m.shape[-1])))
        fut = first_m[:, :, cut:]
        raw_load = fut.pow(2).mean(dim=1).sqrt()
    y_seq = F.interpolate(raw_load.unsqueeze(1), size=horizon, mode="linear",
                          align_corners=False).squeeze(1)
    return x_ctx, y_seq


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Analytical Continuous Ranked Probability Score (CRPS) for Gaussian N(mu, sigma^2)."""
    sig = np.maximum(sigma, 1e-4)
    z = (y - mu) / sig
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    crps = sig * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> float:
    """Empirical energy-form CRPS from Monte Carlo trajectory ensemble (B, M, H)."""
    # Term 1: E|X - y|
    t1 = np.mean(np.abs(samples - y[:, None, :]))
    # Term 2: 0.5 * E|X - X'|
    diffs = np.abs(samples[:, :, None, :] - samples[:, None, :, :])
    t2 = 0.5 * np.mean(diffs)
    return float(t1 - t2)


def compute_forecast_metrics(y_true: np.ndarray, mu_pred: np.ndarray,
                             sigma_pred: np.ndarray, samples_pred=None) -> dict:
    """Compute RMSE, MAE, Ramp-event error, Peak-load error, and CRPS."""
    err = mu_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    # Ramp-event error: error on steep load transitions (|dy/dt| in top 20% quantile)
    dy_true = y_true[:, 1:] - y_true[:, :-1]
    dy_pred = mu_pred[:, 1:] - mu_pred[:, :-1]
    ramp_thr = np.percentile(np.abs(dy_true), 80.0)
    ramp_mask = np.abs(dy_true) >= max(ramp_thr, 1e-6)
    ramp_error = float(np.mean(np.abs(dy_pred[ramp_mask] - dy_true[ramp_mask])))
    # Peak-load error: mean absolute error on per-window maximum load
    peak_true = np.max(y_true, axis=1)
    peak_pred = np.max(mu_pred, axis=1)
    peak_error = float(np.mean(np.abs(peak_pred - peak_true)))
    # CRPS: empirical sample CRPS for cGAN, analytical Gaussian CRPS otherwise
    if samples_pred is not None:
        crps = crps_empirical(samples_pred, y_true)
    else:
        crps = crps_gaussian(mu_pred, sigma_pred, y_true)
    return {
        "rmse": rmse,
        "mae": mae,
        "ramp_error": ramp_error,
        "peak_error": peak_error,
        "crps": crps,
    }


@torch.no_grad()
def predict_forecaster(model, banks, tr, st, hop, horizon=32, bs=256):
    model.eval()
    probs, mus, sigmas, y_trues, samp_list = [], [], [], [], []
    for sel in batches(len(tr), bs, False, None):
        xb = get_batch(banks, tr, st, sel, False, hop)
        x_ctx, y_seq = split_forecast_batch(xb, horizon)
        with amp_ctx():
            out = model(x_ctx)
        probs.append(torch.softmax(out["logits"].float(), 1).cpu().numpy())
        mus.append(out["mu"].float().cpu().numpy())
        sigmas.append(out["sigma"].float().cpu().numpy())
        y_trues.append(y_seq.float().cpu().numpy())
        if "samples" in out:
            samp_list.append(out["samples"].float().cpu().numpy())
    samples = np.concatenate(samp_list) if samp_list else None
    return (np.concatenate(probs), np.concatenate(mus), np.concatenate(sigmas),
            np.concatenate(y_trues), samples)


def train_forecast_fold(args, banks, y, tr_w, va_w, te_w, yt, rng):
    """Train and evaluate AE+LSTM, cGAN, or Multi-task Transformer on one CV fold."""
    in_ch = {m: banks[m].x.shape[1] for m in args.mods}
    fname = args.forecaster
    if fname == "ae_lstm":
        model = AELSTMForecaster(in_ch, horizon=args.horizon, d=args.d).to(DEV)
    elif fname == "mt_transformer":
        model = MultiTaskTransformerForecaster(in_ch, horizon=args.horizon, d=args.d).to(DEV)
    elif fname == "cgan":
        model = CondGANForecaster(in_ch, horizon=args.horizon, d=args.d).to(DEV)
    else:
        raise ValueError(f"Unknown forecaster: {fname}")

    steps = args.epochs * int(np.ceil(len(tr_w[0]) / args.bs))
    if fname == "cgan":
        g_params = (list(model.ctx_enc.parameters()) + list(model.g_fc.parameters()) +
                    list(model.g_net.parameters()))
        d_params = (list(model.d_seq.parameters()) + list(model.d_out.parameters()) +
                    list(model.d_proj.parameters()))
        opt = torch.optim.AdamW(g_params, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
        opt_d = torch.optim.AdamW(d_params, lr=args.lr * 1.5, betas=(0.5, 0.999), weight_decay=args.wd)
    else:
        opt = make_optimizer(args.opt, model.parameters(), args.lr, args.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=steps, pct_start=0.15, anneal_strategy="cos",
            cycle_momentum=args.opt in ("momentum", "nesterov", "adam", "adamw"))

    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        for sel in batches(len(tr_w[0]), args.bs, True, rng):
            xb = get_batch(banks, *tr_w, sel, True, args.hop)
            x_ctx, y_seq = split_forecast_batch(xb, args.horizon)
            yb = yt[torch.from_numpy(tr_w[0][sel]).to(DEV)]
            if fname == "cgan":
                ctx, logits = model.encode_context(x_ctx)
                y_fake = model.sample_trajectories(ctx, n_samples=1).squeeze(1)
                # Discriminator step
                d_real = model.discriminate(ctx.detach(), y_seq)
                d_fake = model.discriminate(ctx.detach(), y_fake.detach())
                loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, 0.9)) +
                          F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                # Generator + context encoder step: adversarial + multi-sample trajectory + peak/ramp + speed cls
                y_samples = model.sample_trajectories(ctx, n_samples=4)
                y_mean = y_samples.mean(dim=1)
                g_adv = F.binary_cross_entropy_with_logits(
                    model.discriminate(ctx, y_samples[:, 0, :]), torch.ones_like(d_real))
                traj_loss = F.mse_loss(y_mean, y_seq) + 0.5 * F.l1_loss(y_mean, y_seq)
                dy_true = y_seq[:, 1:] - y_seq[:, :-1]
                dy_pred = y_mean[:, 1:] - y_mean[:, :-1]
                ramp_l = F.l1_loss(dy_pred, dy_true)
                peak_l = F.l1_loss(y_mean.max(dim=1).values, y_seq.max(dim=1).values)
                cls_l = F.cross_entropy(logits, yb, label_smoothing=args.smooth)
                loss = traj_loss + 0.3 * (ramp_l + peak_l) + 0.1 * g_adv + 0.5 * cls_l
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(g_params, 1.0)
                opt.step()
            else:
                with amp_ctx():
                    out = model(x_ctx)
                mu, sigma, logits = out["mu"].float(), out["sigma"].float(), out["logits"].float()
                # Gaussian NLL + MSE/MAE + explicit Peak & Ramp regularisation + multi-task classification
                nll = 0.5 * (((y_seq - mu) / sigma) ** 2 + 2.0 * torch.log(sigma)).mean()
                traj_l = F.mse_loss(mu, y_seq) + 0.5 * F.l1_loss(mu, y_seq)
                dy_true = y_seq[:, 1:] - y_seq[:, :-1]
                dy_pred = mu[:, 1:] - mu[:, :-1]
                ramp_l = F.l1_loss(dy_pred, dy_true)
                peak_l = F.l1_loss(mu.max(dim=1).values, y_seq.max(dim=1).values)
                cls_l = F.cross_entropy(logits, yb, label_smoothing=args.smooth)
                loss = traj_l + 0.2 * nll + 0.3 * (ramp_l + peak_l) + 0.5 * cls_l
                if fname == "ae_lstm":
                    loss = loss + 0.2 * model.ae_recon_loss(x_ctx)
                elif fname == "mt_transformer" and "peak_pred" in out:
                    aux_peak = F.l1_loss(out["peak_pred"].float(), y_seq.max(dim=1).values)
                    aux_ramp = F.l1_loss(out["ramp_pred"].float(), dy_true.abs().max(dim=1).values)
                    loss = loss + 0.2 * (aux_peak + aux_ramp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

        pv, va_mu, va_sig, va_y, va_samp = predict_forecaster(
            model, banks, *va_w, args.hop, args.horizon)
        va_m = compute_forecast_metrics(va_y, va_mu, va_sig, va_samp)
        va_acc = accuracy_score(y[va_w[0]], pv.argmax(1))
        score = va_m["rmse"] + 0.5 * va_m["crps"] - 0.1 * va_acc
        if score < best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    pt, te_mu, te_sig, te_y, te_samp = predict_forecaster(
        model, banks, *te_w, args.hop, args.horizon)
    f_metrics = compute_forecast_metrics(te_y, te_mu, te_sig, te_samp)
    return pt, te_w, ep + 1, count_params(model), f_metrics


def train_fold(args, keys, data, y, dur, train_t, val_t, test_t, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    banks = {}
    for m in args.mods:
        cat = np.concatenate([data[m][i] for i in train_t])
        mean, std = cat.mean(0), cat.std(0) + 1e-6          # fold-specific normalisation stats
        banks[m] = WindowBank(data[m], mean, std, FS[m], args.win)
    tr_w = make_windows(train_t, dur, args.win, args.hop)
    va_w = make_windows(val_t, dur, args.win, args.hop)
    te_w = make_windows(test_t, dur, args.win, args.hop)
    yt = torch.from_numpy(y).to(DEV)

    if args.task == "forecast":
        pt, te_w, n_ep, n_par, f_metrics = train_forecast_fold(
            args, banks, y, tr_w, va_w, te_w, yt, rng)
        del banks
        empty_device_cache()
        return pt, te_w, n_ep, n_par, f_metrics

    model = FusionNet({m: banks[m].x.shape[1] for m in args.mods}, encoder=args.encoder,
                      d=args.d, p=args.dropout, mod_drop=args.mod_drop).to(DEV)
    extra_metrics = {}
    if args.pretrain != "none":
        reconstructors = pretrain(model, banks, *tr_w, y, args, rng)
        if reconstructors:
            extra_metrics.update(
                eval_reconstruction_and_anomaly(reconstructors, banks, va_w, te_w, y, args))
    G = train_gan(banks, *tr_w, y, args, rng) if (args.gan_aug > 0 or args.gan_only) else None

    opt = make_optimizer(args.opt, model.parameters(), args.lr, args.wd)
    steps = args.epochs * int(np.ceil(len(tr_w[0]) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.15, anneal_strategy="cos",
                                                # cycle momentum/beta1 only for optimizers that
                                                # natively use it, so SGD/Adagrad/RMSProp stay "pure"
                                                cycle_momentum=args.opt in ("momentum", "nesterov",
                                                                            "adam", "adamw"))
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        for sel in batches(len(tr_w[0]), args.bs, True, rng):
            xb = get_batch(banks, *tr_w, sel, True, args.hop)
            yb = yt[torch.from_numpy(tr_w[0][sel]).to(DEV)]
            if G is not None:
                n_syn = len(sel) if args.gan_only else int(round(args.gan_aug * len(sel)))
                ys = torch.randint(0, 3, (n_syn,), device=DEV)
                with torch.no_grad():
                    xs = G(ys)
                if args.gan_only:       # train-on-synthetic, test-on-real (GAN fidelity check)
                    xb, yb = xs, ys
                else:                   # GAN data augmentation: mix synthetic into the batch
                    xb = {m: torch.cat([xb[m], xs[m]]) for m in xb}
                    yb = torch.cat([yb, ys])
            with amp_ctx():
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb, label_smoothing=args.smooth)
            if args.l1 > 0:   # optional L1 penalty (L2 is handled as decoupled weight decay)
                loss = loss + args.l1 * sum(p.abs().sum() for n, p in model.named_parameters()
                                            if p.ndim > 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guards against exploding grads
            opt.step()
            sched.step()
        pv = predict(model, banks, *va_w, args.hop)
        acc = accuracy_score(y[va_w[0]], pv.argmax(1))
        vloss = -np.log(pv[np.arange(len(pv)), y[va_w[0]]] + 1e-9).mean()
        score = acc - 0.01 * vloss
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:       # early stopping
                break
    model.load_state_dict(best_state)
    pt = predict(model, banks, *te_w, args.hop)
    del banks
    empty_device_cache()
    return pt, te_w, ep + 1, count_params(model), extra_metrics


def run(args):
    global DEV
    if args.device != "auto":
        DEV = torch.device(args.device)
    keys, data, subj, y, dur = load_trials(args.mods)
    print(f"{len(keys)} trials, {len(np.unique(subj))} subjects, mods={args.mods}, "
          f"task={args.task}, channels={ {m: data[m][0].shape[1] for m in args.mods} }, device={DEV}")
    folds = subject_folds(subj, args.folds, args.seed)
    all_p, all_t, all_y, all_trial = [], [], [], []
    fold_stats, fold_extras = [], []
    t0 = time.time()
    for f, test_s in enumerate(folds):
        rest = np.setdiff1d(np.unique(subj), test_s)
        rng = np.random.RandomState(args.seed + f)
        val_s = rng.choice(rest, max(1, int(round(len(rest) * 0.15))), replace=False)
        idx = np.arange(len(keys))
        test_t = idx[np.isin(subj, test_s)]
        val_t = idx[np.isin(subj, val_s)]
        train_t = idx[~np.isin(subj, np.concatenate([test_s, val_s]))]
        pt, (tw, _), n_ep, n_par, extra = train_fold(
            args, keys, data, y, dur, train_t, val_t, test_t, args.seed + f)
        wacc = accuracy_score(y[tw], pt.argmax(1))
        tp = np.stack([pt[tw == i].mean(0) for i in test_t])
        tacc = accuracy_score(y[test_t], tp.argmax(1))
        fold_stats.append((wacc, tacc))
        if extra:
            fold_extras.append(extra)
        extra_str = ""
        if "recon_mse" in extra:
            extra_str = (f" | recon MSE {extra['recon_mse']:.4f} MAE {extra['recon_mae']:.4f} "
                         f"| anom AUC {extra['anomaly_auc_roc']:.4f} F1 {extra['anomaly_f1']:.4f}")
        elif "rmse" in extra:
            extra_str = (f" | RMSE {extra['rmse']:.4f} MAE {extra['mae']:.4f} "
                         f"Ramp {extra['ramp_error']:.4f} Peak {extra['peak_error']:.4f} "
                         f"CRPS {extra['crps']:.4f}")
        print(f"  fold {f}: window acc {wacc:.4f} | trial acc {tacc:.4f}{extra_str} "
              f"| epochs {n_ep} | {time.time() - t0:.0f}s")
        all_p.append(pt); all_t.append(tw); all_y.append(y[tw]); all_trial.append((test_t, tp))

    P, T, Y = np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_y)
    tr_idx = np.concatenate([a for a, _ in all_trial])
    tr_p = np.concatenate([b for _, b in all_trial])
    fs = np.array(fold_stats)
    res = {
        "name": args.name, "task": args.task, "mods": args.mods,
        "encoder": args.forecaster if args.task == "forecast" else args.encoder,
        "opt": args.opt, "pretrain": args.pretrain, "gan_aug": args.gan_aug,
        "gan_only": args.gan_only, "win": args.win, "params": n_par,
        "window_acc": float(accuracy_score(Y, P.argmax(1))),
        "window_acc_fold_std": float(fs[:, 0].std()),
        "window_f1": float(f1_score(Y, P.argmax(1), average="macro")),
        "window_auc_roc": safe_auc_roc(Y, P),
        "trial_acc": float(accuracy_score(y[tr_idx], tr_p.argmax(1))),
        "trial_acc_fold_std": float(fs[:, 1].std()),
        "trial_f1": float(f1_score(y[tr_idx], tr_p.argmax(1), average="macro")),
        "trial_auc_roc": safe_auc_roc(y[tr_idx], tr_p),
        "confusion_window": confusion_matrix(Y, P.argmax(1)).tolist(),
        "minutes": (time.time() - t0) / 60,
    }
    if fold_extras:
        for k in fold_extras[0]:
            vals = np.array([fe[k] for fe in fold_extras], dtype=np.float64)
            res[k] = float(np.nanmean(vals))
            res[f"{k}_fold_std"] = float(np.nanstd(vals))
    print(json.dumps({k: v for k, v in res.items() if k != "confusion_window"}))
    RES.mkdir(exist_ok=True)
    (RES / "preds").mkdir(exist_ok=True)
    with open(RES / "results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    np.savez(RES / "preds" / f"{args.name}.npz", trial_keys=np.array(keys)[tr_idx],
             trial_probs=tr_p, trial_y=y[tr_idx], win_keys=np.array(keys)[T], win_probs=P, win_y=Y)
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="classify", choices=["classify", "anomaly", "forecast"],
                    help="classify (default), anomaly (Sparse AE / Conv-AE / BiGAN), or forecast (AE+LSTM / cGAN / MT-Transformer)")
    ap.add_argument("--mods", nargs="+", default=["IMU"])
    ap.add_argument("--encoder", default="resnet", choices=list(ENCODERS))
    ap.add_argument("--forecaster", default=None, choices=["ae_lstm", "cgan", "mt_transformer"],
                    help="Forecasting / multi-task model when --task forecast")
    ap.add_argument("--horizon", type=int, default=32, help="Forecast horizon steps")
    ap.add_argument("--pretrain", default="none",
                    choices=["none", "ae", "dae", "sae", "conv_ae", "vae", "bigan", "bert"])
    ap.add_argument("--pre_epochs", type=int, default=15)
    ap.add_argument("--gan_aug", type=float, default=0.0,
                    help="synthetic samples per real sample in each batch (0 = off)")
    ap.add_argument("--gan_only", action="store_true", help="train on GAN samples only (TSTR)")
    ap.add_argument("--gan_epochs", type=int, default=60)
    ap.add_argument("--opt", default="adamw",
                    choices=["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mod_drop", type=float, default=0.2)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    if a.encoder == "bert" and a.pretrain == "none":
        a.pretrain = "bert"                       # BERT = ViT backbone + masked-token pre-training
    if a.encoder == "conv_ae" and a.pretrain == "none" and a.task == "anomaly":
        a.pretrain = "conv_ae"
    if a.task == "forecast" and a.forecaster is None:
        a.forecaster = a.encoder if a.encoder in ("ae_lstm", "mt_transformer") else "mt_transformer"
    tag = a.forecaster if a.task == "forecast" else a.encoder
    a.name = a.name or f"{'+'.join(a.mods)}_{tag}_{a.opt}" + (
        f"_{a.pretrain}" if a.pretrain != "none" else "")
    return a


if __name__ == "__main__":
    run(parse())
