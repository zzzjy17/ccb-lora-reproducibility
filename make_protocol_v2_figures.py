"""Figures for the 10% labeled protocol-v2 results."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "figures" / "protocol_v2"
OUT.mkdir(parents=True, exist_ok=True)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        return None

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

DATASETS = ["ChnSentiCorp", "TNEWS"]
METHODS = ["lora", "rc_ccb_lora", "dcr_ccb_lora", "full_finetuning", "full_finetuning_ccb"]
LABELS = {
    "lora": "LoRA", "rc_ccb_lora": "RC-CCB-LoRA",
    "dcr_ccb_lora": "DCR-CCB-LoRA", "full_finetuning": "Full fine-tuning",
    "full_finetuning_ccb": "Full fine-tuning + CCB",
}
COLORS = {
    "lora": "#898f98", "rc_ccb_lora": "#4e9b8f",
    "dcr_ccb_lora": "#245f9b", "full_finetuning": "#333c45",
    "full_finetuning_ccb": "#b66d3c",
}
RUNS = pd.read_csv(ROOT / "results" / "protocol_v2_runs.csv")
TRUTH = pd.read_csv(ROOT / "results" / "protocol_v2_pseudolabel_truth_audit.csv")
SEEDS = [13, 42, 2026, 7, 77]


def finish(fig, stem, multi=False):
    fig.canvas.draw()
    if multi:
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT / f"{stem}.alignment.json"),
            overlay_svg=str(OUT / f"{stem}.alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True,
        )
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def panel_letter(ax, value):
    ax.text(-0.14, 1.04, value, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom")


def results_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    y = np.arange(len(METHODS))
    for ax, ds, letter in zip(axes, DATASETS, "ab"):
        sub = RUNS[RUNS.dataset == ds]
        for i, method in enumerate(METHODS):
            values = sub[sub.method == method].macro_f1.to_numpy()
            ax.plot(values, np.full(5, i), "o", ms=2.5, alpha=0.42, color=COLORS[method], zorder=2)
            mean, sd = values.mean(), values.std(ddof=1)
            ax.errorbar(mean, i, xerr=sd, fmt="o", ms=5, capsize=2.3,
                        color=COLORS[method], lw=1.3, zorder=3)
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_yticks(y, [LABELS[m] for m in METHODS])
        ax.invert_yaxis()
        ax.set_xlabel("Macro-F1")
        ax.grid(axis="x", color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[0].set_xlim(0.83, 0.92)
    axes[1].set_xlim(0.22, 0.53)
    finish(fig, "fig1_main_results", multi=True)


def paired_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    comparators = ["lora", "rc_ccb_lora", "full_finetuning", "full_finetuning_ccb"]
    y = np.arange(len(comparators))
    for ax, ds, letter in zip(axes, DATASETS, "ab"):
        sub = RUNS[RUNS.dataset == ds].set_index(["seed", "method"])
        for i, method in enumerate(comparators):
            vals = np.array([
                sub.loc[(seed, "dcr_ccb_lora"), "macro_f1"] - sub.loc[(seed, method), "macro_f1"]
                for seed in [13, 42, 2026, 7, 77]
            ])
            ax.scatter(vals, np.full(5, i), s=11, alpha=0.6, color=COLORS[method], zorder=3)
            ax.plot([vals.min(), vals.max()], [i, i], color=COLORS[method], lw=0.8)
            ax.scatter([vals.mean()], [i], s=33, facecolor="white", edgecolor=COLORS[method],
                       lw=1.2, zorder=4)
        ax.axvline(0, lw=0.8, color="#333c45")
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_yticks(y, [LABELS[m] for m in comparators])
        ax.invert_yaxis()
        ax.set_xlabel("DCR minus comparator: Macro-F1")
        ax.grid(axis="x", color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[0].set_xlim(-0.037, 0.043)
    axes[1].set_xlim(-0.084, 0.245)
    finish(fig, "fig2_paired_effects", multi=True)


def pseudo_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    methods = ["rc_ccb_lora", "dcr_ccb_lora", "full_finetuning_ccb"]
    for ax, ds, letter in zip(axes, DATASETS, "ab"):
        sub = TRUTH[TRUTH.dataset == ds]
        for i, method in enumerate(methods):
            part = sub[sub.method == method]
            if ds == "ChnSentiCorp":
                x = np.full(5, i, dtype=float) + np.linspace(-0.10, 0.10, 5)
                ax.scatter(x, part.pseudo_label_precision, s=24, marker="o",
                           facecolor="white", edgecolor=COLORS[method], lw=1.1, zorder=3)
                ax.scatter([i], [part.pseudo_label_precision.mean()], s=50,
                           color=COLORS[method], marker="D", zorder=4)
            else:
                ax.scatter(part.selected_count, part.pseudo_label_precision,
                           s=24, marker="o", facecolor="white", edgecolor=COLORS[method],
                           lw=1.1, label=LABELS[method], zorder=3)
                ax.scatter([part.selected_count.mean()], [part.pseudo_label_precision.mean()],
                           s=50, color=COLORS[method], marker="D", zorder=4)
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_xlabel("Method" if ds == "ChnSentiCorp" else "Selected pseudo-labels per seed")
        ax.set_ylabel("Precision against source labels")
        ax.grid(color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[0].set_xlim(-0.5, 2.5)
    axes[0].set_xticks(range(3), ["RC", "DCR", "Full FT + CCB"])
    axes[0].set_ylim(0.93, 1.0)
    axes[1].set_xlim(20, 185)
    axes[1].set_ylim(0.76, 0.98)
    finish(fig, "fig3_pseudolabel_truth", multi=True)


def ece_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    y = np.arange(len(METHODS))
    for ax, ds, letter in zip(axes, DATASETS, "ab"):
        sub = RUNS[RUNS.dataset == ds]
        for i, method in enumerate(METHODS):
            values = sub[sub.method == method].ece.to_numpy()
            ax.scatter(values, np.full(5, i), s=9, alpha=0.5, color=COLORS[method])
            ax.errorbar(values.mean(), i, xerr=values.std(ddof=1), fmt="o", ms=5,
                        capsize=2.3, color=COLORS[method], lw=1.2)
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_yticks(y, [LABELS[m] for m in METHODS])
        ax.invert_yaxis()
        ax.set_xlabel("Expected calibration error")
        ax.grid(axis="x", color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[0].set_xlim(0, 0.14)
    axes[1].set_xlim(0, 0.24)
    finish(fig, "fig4_calibration", multi=True)


def confusion_figure(dataset, stem):
    counts = None
    for seed in [13, 42, 2026, 7, 77]:
        path = ROOT / "results" / "runs" / f"{dataset}_0.10_{seed}_dcr_ccb_lora_protocol_v2" / "predictions.npz"
        pred = np.load(path)
        n = 2 if dataset == "ChnSentiCorp" else 15
        current = np.zeros((n, n), dtype=int)
        np.add.at(current, (pred["y_true"], pred["y_pred"]), 1)
        counts = current if counts is None else counts + current
    pd.DataFrame(counts).to_csv(OUT / f"{stem}_counts.csv", index=False, header=False)
    rates = counts / counts.sum(axis=1, keepdims=True)
    pd.DataFrame(rates).to_csv(OUT / f"{stem}_row_rates.csv", index=False, header=False)
    if dataset == "ChnSentiCorp":
        fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
        im = ax.pcolormesh(np.arange(3)-0.5, np.arange(3)-0.5, rates,
                           cmap="Blues", vmin=0, vmax=1, rasterized=False)
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{rates[i,j]:.3f}", ha="center", va="center",
                        color="white" if rates[i,j] > 0.55 else "#202830", fontsize=9)
        ax.set_xticks(range(2), ["0", "1"])
        ax.set_yticks(range(2), ["0", "1"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("ChnSentiCorp", loc="left", fontweight="bold")
        colorbar = fig.colorbar(im, ax=ax, shrink=0.77, label="Row proportion")
        colorbar.solids.set_rasterized(False)
    else:
        fig, ax = plt.subplots(figsize=(5.7, 4.6), constrained_layout=True)
        im = ax.pcolormesh(np.arange(16)-0.5, np.arange(16)-0.5, rates,
                           cmap="Blues", vmin=0, vmax=0.8, rasterized=False)
        ax.set_ylim(14.5, -0.5)
        ax.set_xticks(range(15), [str(i) for i in range(15)])
        ax.set_yticks(range(15), [str(i) for i in range(15)])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("TNEWS", loc="left", fontweight="bold")
        colorbar = fig.colorbar(im, ax=ax, shrink=0.82, label="Row proportion")
        colorbar.solids.set_rasterized(False)
    finish(fig, stem)


def workflow_figure():
    fig, ax = plt.subplots(figsize=(7.1, 4.7), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 8))
    ax.axis("off")

    def box(x, y, w, h, label, fill="#edf2f5", edge="#60717c", size=7):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=0.9))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color="#53616d"))

    ax.text(0.15, 7.65, "a  Model pathway", fontsize=8, fontweight="bold")
    box(0.15, 6.25, 1.35, 0.78, "Input text\ntokens")
    for dx, dy in [(0.13, 0.13), (0.07, 0.07)]:
        ax.add_patch(Rectangle((1.93 + dx, 6.25 + dy), 2.05, 0.78,
                               facecolor="#d9e7f0", edgecolor="#60717c", lw=0.7))
    box(1.93, 6.25, 2.05, 0.78, "Chinese BERT\nLoRA query/value, r = 8")
    box(4.42, 6.25, 1.15, 0.78, "CLS vector\nh")
    box(6.0, 6.85, 2.1, 0.72, "Shared linear + low-rank\nlogits s", "#e7f2ef")
    box(6.0, 5.75, 2.1, 0.72, "Class-indexed low-rank\nupdate delta, r = 4", "#e7f2ef")
    box(8.45, 6.85, 1.58, 0.72, "Gate u = 1 - c\nc = max softmax(s/T)", "#f1e9da", size=5.9)
    box(10.35, 6.25, 1.5, 0.78, "Final logits\ns + u * delta", "#e7f2ef")
    arrow(1.5, 6.64, 1.93, 6.64)
    arrow(4.05, 6.64, 4.42, 6.64)
    arrow(5.57, 6.64, 6.0, 7.18)
    arrow(5.57, 6.64, 6.0, 6.11)
    arrow(8.1, 7.2, 8.45, 7.2)
    arrow(10.03, 7.2, 10.35, 6.79)
    arrow(8.1, 6.11, 10.35, 6.48)

    ax.text(0.15, 5.1, "b  Training and selection", fontsize=8, fontweight="bold")
    box(0.15, 3.85, 1.65, 0.8, "10% labeled\ntraining subset")
    box(2.2, 3.85, 1.55, 0.8, "Supervised\nwarm start")
    box(4.15, 3.85, 1.75, 0.8, "Separate validation\nfit temperature T", "#f1e9da")
    box(6.3, 3.85, 1.5, 0.8, "Unlabeled pool\nmodel scores")
    box(8.2, 3.85, 1.65, 0.8, "Confidence, margin\nconsistency, spread", "#f7ede6", size=6.7)
    box(10.25, 3.85, 1.6, 0.8, "Class thresholds\nand budgets", "#f7ede6")
    for left, right in [(1.8, 2.2), (3.75, 4.15), (5.9, 6.3), (7.8, 8.2), (9.85, 10.25)]:
        arrow(left, 4.25, right, 4.25)
    box(2.2, 2.4, 2.4, 0.77, "Original labels: weight 1\nSelected labels: weight 0.5", "#f7ede6")
    box(5.05, 2.4, 2.5, 0.77, "Weighted second stage\nfinal checkpoint", "#f7ede6")
    box(8.0, 2.4, 3.0, 0.77, "Held-out evaluation\nMacro-F1, ECE, predictions")
    ax.plot([11.05, 11.05, 3.4], [3.85, 3.52, 3.52],
            color="#53616d", lw=0.9)
    arrow(3.4, 3.52, 3.4, 3.17)
    arrow(4.6, 2.79, 5.05, 2.79)
    arrow(7.55, 2.79, 8.0, 2.79)
    ax.text(0.15, 1.8, "c  Post-training label audit", fontsize=8, fontweight="bold")
    box(2.2, 0.65, 2.4, 0.75, "Saved selections\nand source indices")
    box(5.05, 0.65, 2.5, 0.75, "Original training labels\nwithheld from selector")
    box(8.0, 0.65, 3.0, 0.75, "Selection count and\nsource-label precision")
    arrow(4.6, 1.02, 5.05, 1.02)
    arrow(7.55, 1.02, 8.0, 1.02)
    finish(fig, "fig7_workflow")


def seed_trajectories_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, "ab"):
        sub = RUNS[RUNS.dataset == dataset].set_index(["seed", "method"])
        for method in METHODS:
            values = [sub.loc[(seed, method), "macro_f1"] for seed in SEEDS]
            ax.plot(range(5), values, marker="o", ms=3.4, lw=1.25,
                    color=COLORS[method], label=LABELS[method])
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xticks(range(5), [str(seed) for seed in SEEDS])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Macro-F1")
        ax.grid(axis="y", color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[0].set_ylim(0.83, 0.93)
    axes[1].set_ylim(0.18, 0.53)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.5,
                   ncol=1, frameon=False, handlelength=1.0, handletextpad=1.1)
    finish(fig, "fig8_seed_trajectories", multi=True)


def class_selection_figure():
    from audit_pseudolabel_truth_v2 import source_labels
    labels = source_labels("TNEWS")
    selected = defaultdict(lambda: np.zeros((15, 2), dtype=int))
    for method in ["rc_ccb_lora", "dcr_ccb_lora", "full_finetuning_ccb"]:
        for seed in SEEDS:
            path = ROOT / "results" / "runs" / f"TNEWS_0.10_{seed}_{method}_protocol_v2" / "pseudo_label_audit.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["selected"]:
                    predicted = int(row["predicted_class"])
                    correct = int(predicted == labels[int(row["source_index"])])
                    selected[method][predicted, 0 if correct else 1] += 1
    rows = []
    for method, arr in selected.items():
        for cls in range(15):
            rows.append({"method": method, "predicted_class": cls,
                         "correct": arr[cls, 0], "incorrect": arr[cls, 1]})
    pd.DataFrame(rows).to_csv(OUT / "fig9_tnews_class_selection.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    x = np.arange(15)
    width = 0.25
    for i, method in enumerate(["rc_ccb_lora", "dcr_ccb_lora", "full_finetuning_ccb"]):
        arr = selected[method]
        offset = x + (i - 1) * width
        ax.bar(offset, arr[:, 0], width=width, color=COLORS[method],
               edgecolor="white", linewidth=0.3, label=LABELS[method])
        ax.bar(offset, arr[:, 1], bottom=arr[:, 0], width=width,
               color=COLORS[method], alpha=0.35, edgecolor="#333c45",
               linewidth=0.3)
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("Predicted TNEWS class")
    ax.set_ylabel("Selected pseudo-labels across five runs")
    ax.set_title("Correct selections (dark) and errors (light)", loc="left", fontweight="bold")
    ax.set_ylim(0, max((arr.sum(axis=1).max() for arr in selected.values())) * 1.12)
    ax.grid(axis="y", color="#e3e7e9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.3)
    finish(fig, "fig9_tnews_class_selection")


def reliability_figure():
    methods = ["lora", "dcr_ccb_lora", "full_finetuning"]
    rows = []
    for dataset in DATASETS:
        for method in methods:
            probs, truths = [], []
            for seed in SEEDS:
                path = ROOT / "results" / "runs" / f"{dataset}_0.10_{seed}_{method}_protocol_v2" / "predictions.npz"
                with np.load(path) as z:
                    probs.append(z["probabilities"])
                    truths.append(z["y_true"])
            p = np.concatenate(probs)
            y = np.concatenate(truths)
            confidence = p.max(1)
            correct = (p.argmax(1) == y).astype(float)
            for index in range(10):
                low, high = index / 10, (index + 1) / 10
                mask = (confidence > low) & (confidence <= high)
                rows.append({"dataset": dataset, "method": method, "bin": index,
                             "count": int(mask.sum()),
                             "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                             "empirical_accuracy": float(correct[mask].mean()) if mask.any() else np.nan})
    data = pd.DataFrame(rows)
    data.to_csv(OUT / "fig10_reliability_bins.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, "ab"):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#a1a9ae")
        for method in methods:
            part = data[(data.dataset == dataset) & (data.method == method) & (data["count"] > 0)]
            ax.plot(part.mean_confidence, part.empirical_accuracy,
                    lw=1.2, color=COLORS[method], label=LABELS[method])
            ax.scatter(part.mean_confidence, part.empirical_accuracy,
                       s=np.clip(part["count"].to_numpy() / 250, 9, 55),
                       color=COLORS[method], zorder=3)
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Observed accuracy in bin")
        ax.grid(color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.2,
                   handlelength=1.0, handletextpad=1.1)
    finish(fig, "fig10_reliability", multi=True)


def runtime_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, "ab"):
        sub = RUNS[RUNS.dataset == dataset]
        x = np.arange(5)
        means = [sub[sub.method == method].train_seconds.mean() for method in METHODS]
        ax.bar(x, means, color=[COLORS[m] for m in METHODS], width=0.55)
        for i, method in enumerate(METHODS):
            vals = sub[sub.method == method].train_seconds.to_numpy()
            ax.scatter(np.full(5, i), vals, s=7, facecolors="white", edgecolors="#333c45",
                       linewidth=0.5, zorder=3)
        ax.set_xticks(x, ["LoRA", "RC", "DCR", "Full FT", "Full FT\n+ CCB"])
        ax.set_ylabel("Observed run time (s)")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    finish(fig, "fig11_run_time", multi=True)


def selection_reasons_figure():
    methods = ["rc_ccb_lora", "dcr_ccb_lora", "full_finetuning_ccb"]
    reasons = ["class_budget", "confidence", "selected", "other"]
    shades = {"class_budget": "#899aa7", "confidence": "#245f9b",
              "selected": "#4e9b8f", "other": "#b66d3c"}
    rows = []
    for dataset in DATASETS:
        for method in methods:
            tally = Counter()
            for seed in SEEDS:
                path = ROOT / "results" / "runs" / f"{dataset}_0.10_{seed}_{method}_protocol_v2" / "pseudo_label_audit.jsonl"
                tally.update(json.loads(line)["reason"] for line in path.read_text(encoding="utf-8").splitlines())
            for reason in reasons:
                count = tally[reason] if reason != "other" else sum(v for k, v in tally.items() if k not in reasons)
                rows.append({"dataset": dataset, "method": method, "reason": reason,
                             "count": count, "candidate_decisions": sum(tally.values())})
    pd.DataFrame(rows).to_csv(OUT / "fig12_selection_reasons.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, "ab"):
        part = pd.DataFrame(rows).query("dataset == @dataset")
        y = np.arange(3)
        left = np.zeros(3)
        for reason in reasons:
            counts = np.array([part.loc[(part.method == method) & (part.reason == reason), "count"].iloc[0]
                               for method in methods])
            total = np.array([part.loc[part.method == method, "candidate_decisions"].iloc[0]
                              for method in methods])
            values = counts / total * 100
            ax.barh(y, values, left=left, height=0.52, color=shades[reason], label=reason.replace("_", " "))
            left += values
        ax.set_yticks(y, ["RC", "DCR", "Full FT + CCB"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of candidate decisions (%)")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e3e7e9", lw=0.6)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.0, frameon=False)
    finish(fig, "fig12_selection_reasons", multi=True)


def class_recall_figure():
    methods = ["rc_ccb_lora", "dcr_ccb_lora", "full_finetuning"]
    rows = []
    for method in methods:
        totals = np.zeros(15, dtype=int)
        correct = np.zeros(15, dtype=int)
        for seed in SEEDS:
            path = ROOT / "results" / "runs" / f"TNEWS_0.10_{seed}_{method}_protocol_v2" / "predictions.npz"
            with np.load(path) as z:
                truth = z["y_true"].astype(int)
                predicted = z["probabilities"].argmax(axis=1)
            totals += np.bincount(truth, minlength=15)
            correct += np.bincount(truth[truth == predicted], minlength=15)
        for cls in range(15):
            rows.append({"method": method, "class": cls, "correct": correct[cls],
                         "evaluations": totals[cls], "recall": correct[cls] / totals[cls]})
    data = pd.DataFrame(rows)
    data.to_csv(OUT / "fig13_tnews_class_recall.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.0), constrained_layout=True)
    x = np.arange(15)
    for method in methods:
        part = data[data.method == method]
        ax.plot(x, part.recall, marker="o", ms=3, lw=1.2,
                color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("True TNEWS class")
    ax.set_ylabel("Recall across five evaluation runs")
    ax.set_title("Class-wise recall exposes uneven multiclass performance", loc="left", fontweight="bold")
    ax.grid(color="#e3e7e9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.0, frameon=False)
    finish(fig, "fig13_tnews_class_recall")


if __name__ == "__main__":
    results_figure()
    paired_figure()
    pseudo_figure()
    ece_figure()
    confusion_figure("ChnSentiCorp", "fig5_confusion_chnsenticorp")
    confusion_figure("TNEWS", "fig6_confusion_tnews")
    workflow_figure()
    seed_trajectories_figure()
    class_selection_figure()
    reliability_figure()
    runtime_figure()
    selection_reasons_figure()
    class_recall_figure()
    print(f"Figures written to {OUT}")
