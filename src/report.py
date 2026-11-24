"""Summarise all runs, build a soft-voting ensemble, and draw figures into results/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RES = Path(__file__).resolve().parents[1] / "results"
LABELS = ["0.5 m/s", "0.75 m/s", "1.0 m/s"]


def latest_runs():
    runs = {}
    for line in open(RES / "results.jsonl"):
        r = json.loads(line)
        runs[r["name"]] = r
    return runs


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
    rows = sorted(runs.values(), key=lambda r: -r["window_acc"])
    lines = ["| Run | Modalities | Encoder | Optimizer | Params | Window acc | Window F1 | Trial acc |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {'+'.join(r['mods'])} | {r['encoder']} | {r['opt']} | "
                     f"{r['params']/1e6:.2f}M | {100*r['window_acc']:.2f} ± {100*r['window_acc_fold_std']:.2f} | "
                     f"{100*r['window_f1']:.2f} | {100*r['trial_acc']:.2f} |")
    out = ["## All runs (5-fold subject-independent CV)", "", *lines, ""]

    if ens_members:
        members = [m for m in ens_members if m in runs]
        try:
            p, y = ensemble(members, "win")
            wacc, wf1 = accuracy_score(y, p.argmax(1)), f1_score(y, p.argmax(1), average="macro")
        except AssertionError:
            wacc = wf1 = float("nan")
        tp, ty = ensemble(members, "trial")
        tacc = accuracy_score(ty, tp.argmax(1))
        out += ["## Soft-voting ensemble", "", f"Members: {', '.join(members)}", "",
                f"- Window accuracy: **{100*wacc:.2f}%** (macro-F1 {100*wf1:.2f}%)",
                f"- Trial accuracy: **{100*tacc:.2f}%**", ""]
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
