"""Subject-independent walking-speed classification (0.5 / 0.75 / 1.0 m/s).

Evaluation protocol: K-fold cross-validation grouped by SUBJECT, so no person in a test fold
was seen during training. Inside each training fold ~15% of subjects are held out for early
stopping / model selection. Metrics are reported per window and per trial (averaging the
softmax over every window of a 60-s recording).

Example:
  python src/train.py --mods IMU FP EMG --encoder resnet --epochs 40
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models import (ENCODERS, AutoEncoder, CondDiscriminator, CondGenerator, FusionNet,
                    count_params)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FS = {"EMG": 16, "FP": 100, "IMU": 60, "EEG": 100}
SPEEDS = ["0.5", "0.75", "1"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data
def load_trials(mods):
    keys = None
    for m in mods:
        ks = {p.stem for p in (PROC / m).glob("S*.npy")}
        keys = ks if keys is None else keys & ks
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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            logits = model(get_batch(banks, tr, st, sel, False, hop))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(probs)


def pretrain(model, banks, tr, st, args, rng):
    """Unsupervised pre-training of each modality encoder on the TRAINING windows only
    (labels unused): AE / denoising AE / sparse AE / VAE reconstruction, or BERT masked-token
    modelling. The pre-trained encoder weights are then fine-tuned with labels."""
    for m, b in banks.items():
        enc = model.enc[m]
        if args.pretrain == "bert":
            obj, loss_fn = enc, enc.mlm_loss
        else:
            obj = AutoEncoder(enc, b.x.shape[1], b.L, args.pretrain, args.d).to(DEV)
            loss_fn = obj
        opt = torch.optim.AdamW(obj.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(args.pre_epochs):
            obj.train()
            tot = 0
            for sel in batches(len(tr), args.bs, True, rng):
                x = get_batch({m: b}, tr, st, sel, True, args.hop)[m]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                    loss = loss_fn(x)
                opt.zero_grad(set_to_none=True)
                loss.float().backward()
                nn.utils.clip_grad_norm_(obj.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(sel)
        print(f"    {args.pretrain.upper()}[{m}] pre-train loss {tot / len(tr):.4f}")


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

    model = FusionNet({m: banks[m].x.shape[1] for m in args.mods}, encoder=args.encoder,
                      d=args.d, p=args.dropout, mod_drop=args.mod_drop).to(DEV)
    if args.pretrain != "none":
        pretrain(model, banks, *tr_w, args, rng)
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
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
    torch.cuda.empty_cache()
    return pt, te_w, ep + 1, count_params(model)


def run(args):
    keys, data, subj, y, dur = load_trials(args.mods)
    print(f"{len(keys)} trials, {len(np.unique(subj))} subjects, mods={args.mods}, "
          f"channels={ {m: data[m][0].shape[1] for m in args.mods} }")
    folds = subject_folds(subj, args.folds, args.seed)
    all_p, all_t, all_y, all_trial = [], [], [], []
    fold_stats = []
    t0 = time.time()
    for f, test_s in enumerate(folds):
        rest = np.setdiff1d(np.unique(subj), test_s)
        rng = np.random.RandomState(args.seed + f)
        val_s = rng.choice(rest, max(1, int(round(len(rest) * 0.15))), replace=False)
        idx = np.arange(len(keys))
        test_t = idx[np.isin(subj, test_s)]
        val_t = idx[np.isin(subj, val_s)]
        train_t = idx[~np.isin(subj, np.concatenate([test_s, val_s]))]
        pt, (tw, _), n_ep, n_par = train_fold(args, keys, data, y, dur, train_t, val_t, test_t,
                                              args.seed + f)
        wacc = accuracy_score(y[tw], pt.argmax(1))
        tp = np.stack([pt[tw == i].mean(0) for i in test_t])
        tacc = accuracy_score(y[test_t], tp.argmax(1))
        fold_stats.append((wacc, tacc))
        print(f"  fold {f}: window acc {wacc:.4f} | trial acc {tacc:.4f} | epochs {n_ep} "
              f"| {time.time() - t0:.0f}s")
        all_p.append(pt); all_t.append(tw); all_y.append(y[tw]); all_trial.append((test_t, tp))

    P, T, Y = np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_y)
    tr_idx = np.concatenate([a for a, _ in all_trial])
    tr_p = np.concatenate([b for _, b in all_trial])
    fs = np.array(fold_stats)
    res = {
        "name": args.name, "mods": args.mods, "encoder": args.encoder, "opt": args.opt,
        "pretrain": args.pretrain, "gan_aug": args.gan_aug, "gan_only": args.gan_only,
        "win": args.win, "params": n_par,
        "window_acc": float(accuracy_score(Y, P.argmax(1))),
        "window_acc_fold_std": float(fs[:, 0].std()),
        "window_f1": float(f1_score(Y, P.argmax(1), average="macro")),
        "trial_acc": float(accuracy_score(y[tr_idx], tr_p.argmax(1))),
        "trial_acc_fold_std": float(fs[:, 1].std()),
        "trial_f1": float(f1_score(y[tr_idx], tr_p.argmax(1), average="macro")),
        "confusion_window": confusion_matrix(Y, P.argmax(1)).tolist(),
        "minutes": (time.time() - t0) / 60,
    }
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
    ap.add_argument("--mods", nargs="+", default=["IMU"])
    ap.add_argument("--encoder", default="resnet", choices=list(ENCODERS))
    ap.add_argument("--pretrain", default="none", choices=["none", "ae", "dae", "sae", "vae", "bert"])
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
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    if a.encoder == "bert" and a.pretrain == "none":
        a.pretrain = "bert"                       # BERT = ViT backbone + masked-token pre-training
    a.name = a.name or f"{'+'.join(a.mods)}_{a.encoder}_{a.opt}" + (
        f"_{a.pretrain}" if a.pretrain != "none" else "")
    return a


if __name__ == "__main__":
    run(parse())
