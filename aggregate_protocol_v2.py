"""Validate and summarize the independent-calibration protocol runs."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "results" / "runs"
RESULTS = ROOT / "results"
DATASETS = ("ChnSentiCorp", "TNEWS")
METHODS = (
    "lora", "rc_ccb_lora", "dcr_ccb_lora", "full_finetuning",
    "full_finetuning_ccb",
)
SEEDS = (13, 42, 2026, 7, 77)


def main():
    rows = []
    manifests = {}
    code_hashes = set()
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                run_dir = RUN_ROOT / f"{dataset}_0.10_{seed}_{method}_protocol_v2"
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                assert metrics["dataset"] == dataset and metrics["seed"] == seed
                assert metrics["method"] == method and metrics["label_fraction"] == 0.1
                assert metrics["protocol_version"] == manifest["protocol_version"]
                assert metrics["code_sha256"] == manifest["code_sha256"]
                assert metrics["device"] == "cuda"
                assert metrics["n_eval"] == metrics["evaluation_size"]
                assert not (set(manifest["labeled_indices"]) & set(manifest["unlabeled_indices"]))
                if dataset == "TNEWS":
                    assert len(manifest["calibration_indices"]) == 5000
                    assert len(manifest["evaluation_indices"]) == 5000
                    assert not (set(manifest["calibration_indices"]) & set(manifest["evaluation_indices"]))
                    assert metrics["calibration_source"] == "TNEWS_dev_half_by_seed"
                if metrics["pseudo_label_count"]:
                    audit_file = run_dir / "pseudo_label_audit.jsonl"
                    audit = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
                    selected = [row for row in audit if row["selected"]]
                    assert len(selected) == metrics["pseudo_label_count"]
                    for row in selected:
                        assert row["index"] in manifest["unlabeled_indices"]
                        assert row["source_index"] == manifest["source_train_indices"][row["index"]]
                code_hashes.add(metrics["code_sha256"])
                manifests[(dataset, seed, method)] = manifest
                rows.append(metrics)
    assert len(rows) == 50
    assert len(code_hashes) == 1, code_hashes
    for dataset in DATASETS:
        for seed in SEEDS:
            first = manifests[(dataset, seed, METHODS[0])]
            for method in METHODS[1:]:
                other = manifests[(dataset, seed, method)]
                for key in ("labeled_indices", "unlabeled_indices", "calibration_indices", "evaluation_indices"):
                    assert first[key] == other[key], (dataset, seed, method, key)

    runs = pd.DataFrame(rows)
    cols = ["dataset", "seed", "method", "macro_f1", "accuracy", "ece", "brier_score",
            "pseudo_label_count", "train_seconds", "n_labeled", "n_unlabeled", "n_eval",
            "calibration_size", "code_sha256", "protocol_version"]
    runs[cols].sort_values(["dataset", "seed", "method"]).to_csv(
        RESULTS / "protocol_v2_runs.csv", index=False
    )
    summary = runs.groupby(["dataset", "method"], as_index=False).agg(
        n=("macro_f1", "size"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        ece_mean=("ece", "mean"),
        ece_sd=("ece", "std"),
        pseudo_count_mean=("pseudo_label_count", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
    )
    summary.to_csv(RESULTS / "protocol_v2_summary.csv", index=False)

    differences = []
    for dataset in DATASETS:
        subset = runs[runs.dataset == dataset].set_index(["seed", "method"])
        for method in METHODS:
            if method == "dcr_ccb_lora":
                continue
            paired = np.array([
                subset.loc[(seed, "dcr_ccb_lora"), "macro_f1"]
                - subset.loc[(seed, method), "macro_f1"]
                for seed in SEEDS
            ])
            differences.append({
                "dataset": dataset, "contrast": f"dcr_ccb_lora - {method}",
                "n_pairs": len(paired), "mean_delta_macro_f1": paired.mean(),
                "sd_delta_macro_f1": paired.std(ddof=1),
                "min_delta_macro_f1": paired.min(), "max_delta_macro_f1": paired.max(),
                "positive_pairs": int((paired > 0).sum()),
                "paired_deltas_by_seed": json.dumps(dict(zip(SEEDS, paired.tolist()))),
            })
    pd.DataFrame(differences).to_csv(RESULTS / "protocol_v2_paired_differences.csv", index=False)

    truth = pd.read_csv(RESULTS / "protocol_v2_pseudolabel_truth_audit.csv")
    assert len(truth) == 30
    truth_summary = truth.groupby(["dataset", "method"], as_index=False).agg(
        n=("seed", "size"), selected_total=("selected_count", "sum"),
        correct_total=("correct_count", "sum"),
        precision_mean=("pseudo_label_precision", "mean"),
        precision_sd=("pseudo_label_precision", "std"),
    )
    truth_summary["pooled_precision"] = truth_summary.correct_total / truth_summary.selected_total
    truth_summary.to_csv(RESULTS / "protocol_v2_pseudolabel_truth_summary.csv", index=False)
    print("Validated 50 runs, 30 pseudo-label audits, one code hash, and disjoint TNEWS splits.")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Paired DCR contrasts:")
    print(pd.DataFrame(differences)[["dataset", "contrast", "mean_delta_macro_f1", "positive_pairs"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Pseudo-label truth:")
    print(truth_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
