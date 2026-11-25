"""Summarise all runs, build a soft-voting ensemble, and draw figures into results/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

RES = Path(__file__).resolve().parents[1] / "results"
LABELS = ["0.5 m/s", "0.75 m/s", "1.0 m/s"]


def latest_runs():
    runs = {}
    if not (RES / "results.jsonl").exists():
        return runs
    for line in open(RES / "results.jsonl"):
        r = json.loads(line)
        runs[r["name"]] = r
    return runs


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except Exception:  # noqa: BLE001
        return float("nan")


def ensemble(names, level="win"):
    """Average class probabilities of several runs over the samples they share."""
    preds = [np.load(RES / "preds" / f"{n}.npz") for n in names]
    if level == "trial":
        common = set(preds[0]["trial_keys"])
        for p in preds[1:]:
            common &= set(p["trial_keys"])
        common = sorted(common)
        probs, y = 0, None
        for p in preds:
            pos = {k: i for i, k in enumerate(p["trial_keys"])}
            ix = [pos[k] for k in common]
            probs = probs + p["trial_probs"][ix]
            y = p["trial_y"][ix]
        return probs / len(preds), y
    # window level: only valid when runs share an identical window list
    base = preds[0]
    for p in preds[1:]:
        assert np.array_equal(p["win_keys"], base["win_keys"]), "window lists differ"
    return sum(p["win_probs"] for p in preds) / len(preds), base["win_y"]


def main(ens_members=None):
    runs = latest_runs()
    if not runs:
        print("No completed runs found in results/results.jsonl yet.")
        return
    rows = sorted(runs.values(), key=lambda r: -r["window_acc"])
    lines = ["| Run | Modalities | Encoder | Optimizer | Params | Window acc | Window F1 | Window AUC-ROC | Trial acc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        w_auc = r.get("window_auc_roc")
        auc_str = f"{100*w_auc:.2f}" if w_auc is not None and w_auc == w_auc else "—"
        lines.append(f"| {r['name']} | {'+'.join(r['mods'])} | {r['encoder']} | {r['opt']} | "
                     f"{r['params']/1e6:.2f}M | {100*r['window_acc']:.2f} ± {100*r['window_acc_fold_std']:.2f} | "
                     f"{100*r['window_f1']:.2f} | {auc_str} | {100*r['trial_acc']:.2f} |")
    out = ["## All runs (5-fold subject-independent CV)", "", *lines, ""]

    # Suite E: Reconstruction & Anomaly Detection (Sparse AE, Conv-AE, BiGAN, etc.)
    recon_rows = [r for r in rows if "recon_mse" in r]
    if recon_rows:
        rlines = [
            "| Run | Model / Pretrain | Modalities | Recon MSE | Recon MAE | Anomaly AUC-ROC | Anomaly F1 | Class AUC-ROC | Class F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(recon_rows, key=lambda x: x["recon_mse"]):
            w_auc = r.get("window_auc_roc", float("nan"))
            rlines.append(
                f"| {r['name']} | {r['encoder']} ({r.get('pretrain', 'none')}) | {'+'.join(r['mods'])} | "
                f"{r['recon_mse']:.4f} ± {r.get('recon_mse_fold_std', 0.0):.4f} | "
                f"{r['recon_mae']:.4f} ± {r.get('recon_mae_fold_std', 0.0):.4f} | "
                f"{100*r.get('anomaly_auc_roc', 0.0):.2f} | {100*r.get('anomaly_f1', 0.0):.2f} | "
                f"{100*w_auc:.2f} | {100*r['window_f1']:.2f} |"
            )
        out += ["## Unsupervised Reconstruction & Anomaly Detection (Sparse AE, Conv-AE, BiGAN)", "", *rlines, ""]

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        rnames = [r["name"] for r in recon_rows]
        x = np.arange(len(rnames))
        axes[0].bar(x - 0.18, [r["recon_mse"] for r in recon_rows], 0.35, label="Recon MSE", color="#4C72B0")
        axes[0].bar(x + 0.18, [r["recon_mae"] for r in recon_rows], 0.35, label="Recon MAE", color="#DD8452")
        axes[0].set_xticks(x, rnames, rotation=25, ha="right")
        axes[0].set_ylabel("Error"); axes[0].set_title("Reconstruction Error (MSE / MAE)"); axes[0].legend()
        axes[1].bar(x - 0.18, [100 * r.get("anomaly_auc_roc", 0.0) for r in recon_rows], 0.35, label="AUC-ROC (%)", color="#55A868")
        axes[1].bar(x + 0.18, [100 * r.get("anomaly_f1", 0.0) for r in recon_rows], 0.35, label="F1 (%)", color="#C44E52")
        axes[1].set_xticks(x, rnames, rotation=25, ha="right")
        axes[1].set_ylim(0, 105); axes[1].set_ylabel("Score (%)"); axes[1].set_title("Anomaly Detection (AUC-ROC / F1)"); axes[1].legend()
        fig.tight_layout(); fig.savefig(RES / "reconstruction_anomaly.png", dpi=150); plt.close(fig)

    # Suite F: Continuous Load Forecasting & Multi-Task Probabilistic Models (AE+LSTM, cGAN, MT-Transformer)
    fcast_rows = [r for r in rows if "rmse" in r]
    if fcast_rows:
        flines = [
            "| Run | Model | Modalities | RMSE | MAE | Ramp-event error | Peak-load error | CRPS | Window acc | Window F1 | Window AUC-ROC |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(fcast_rows, key=lambda x: x["crps"]):
            w_auc = r.get("window_auc_roc", float("nan"))
            flines.append(
                f"| {r['name']} | {r['encoder']} | {'+'.join(r['mods'])} | "
                f"{r['rmse']:.4f} ± {r.get('rmse_fold_std', 0.0):.4f} | "
                f"{r['mae']:.4f} ± {r.get('mae_fold_std', 0.0):.4f} | "
                f"{r['ramp_error']:.4f} ± {r.get('ramp_error_fold_std', 0.0):.4f} | "
                f"{r['peak_error']:.4f} ± {r.get('peak_error_fold_std', 0.0):.4f} | "
                f"{r['crps']:.4f} ± {r.get('crps_fold_std', 0.0):.4f} | "
                f"{100*r['window_acc']:.2f} | {100*r['window_f1']:.2f} | {100*w_auc:.2f} |"
            )
        out += ["## Load Forecasting & Multi-Task Probabilistic Models (AE + LSTM, cGAN, Multi-task Transformer)", "", *flines, ""]

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        fnames = [r["name"] for r in fcast_rows]
        metrics = [("rmse", "RMSE"), ("mae", "MAE"), ("ramp_error", "Ramp Err"),
                   ("peak_error", "Peak Err"), ("crps", "CRPS")]
        x = np.arange(len(fnames))
        w = 0.15
        for idx_m, (mk, mlabel) in enumerate(metrics):
            ax.bar(x + (idx_m - 2) * w, [r[mk] for r in fcast_rows], w, label=mlabel)
        ax.set_xticks(x, fnames, rotation=15, ha="right")
        ax.set_ylabel("Error / Score (normalised load units)")
        ax.set_title("Forecasting & Probabilistic Metrics (5-fold CV)")
        ax.legend()
        fig.tight_layout(); fig.savefig(RES / "forecasting_metrics.png", dpi=150); plt.close(fig)

    if ens_members:
        members = [m for m in ens_members if m in runs]
        if members:
            try:
                p, y = ensemble(members, "win")
                wacc, wf1, wauc = accuracy_score(y, p.argmax(1)), f1_score(y, p.argmax(1), average="macro"), safe_auc(y, p)
            except AssertionError:
                wacc = wf1 = wauc = float("nan")
            tp, ty = ensemble(members, "trial")
            tacc, tf1, tauc = accuracy_score(ty, tp.argmax(1)), f1_score(ty, tp.argmax(1), average="macro"), safe_auc(ty, tp)
            out += ["## Soft-voting ensemble", "", f"Members: {', '.join(members)}", "",
                    f"- Window accuracy: **{100*wacc:.2f}%** (macro-F1 {100*wf1:.2f}%, AUC-ROC {100*wauc:.2f}%)",
                    f"- Trial accuracy: **{100*tacc:.2f}%** (macro-F1 {100*tf1:.2f}%, AUC-ROC {100*tauc:.2f}%)", ""]
            cm = confusion_matrix(y, p.argmax(1)) if wacc == wacc else confusion_matrix(ty, tp.argmax(1))
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
            ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Ensemble confusion matrix")
            fig.tight_layout(); fig.savefig(RES / "confusion_ensemble.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(rows) + 1))
    names = [r["name"] for r in rows][::-1]
    vals = [100 * r["window_acc"] for r in rows][::-1]
    errs = [100 * r["window_acc_fold_std"] for r in rows][::-1]
    ax.barh(names, vals, xerr=errs, color="#4C72B0")
    ax.set_xlabel("Window accuracy (%) — subject-independent 5-fold CV")
    ax.set_xlim(max(0, min(vals) - 10), 100)
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "accuracy_by_run.png", dpi=150); plt.close(fig)

    (RES / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out).encode("ascii", "replace").decode())


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
