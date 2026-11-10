"""Post-hoc pseudo-label audit using labels hidden during model training.

The source labels are read only after a completed run. They are never passed
to calibration, selection, or retraining.
"""
from ast import literal_eval
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "results" / "runs"
OUT = ROOT / "results" / "protocol_v2_pseudolabel_truth_audit.csv"


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def source_labels(dataset):
    if dataset == "ChnSentiCorp":
        rows = read_jsonl(ROOT / "data" / "chnsenticorp" / "train.jsonl")
        return np.asarray([int(r["label"]) for r in rows])
    if dataset == "TNEWS":
        data = ROOT / "data" / "tnews"
        label_rows = [json.loads(x) for x in (data / "labels.json").read_text(encoding="utf-8").splitlines()]
        mapping = {int(r["label"]): i for i, r in enumerate(label_rows)}
        rows = [json.loads(x) for x in (data / "train.json").read_text(encoding="utf-8").splitlines() if x.strip()]
        return np.asarray([mapping[int(r["label"])] for r in rows])
    if dataset == "IFLYTEK":
        data = ROOT / "data" / "iflytek"
        rows = read_jsonl(data / "train.json")
        label_ids = sorted({int(r["label"]) for r in rows})
        mapping = {label: i for i, label in enumerate(label_ids)}
        return np.asarray([mapping[int(r["label"])] for r in rows])
    raise ValueError(dataset)


def main():
    records = []
    for audit_path in sorted(RUN_ROOT.glob("*_protocol_v2/pseudo_label_audit.jsonl")):
        run_dir = audit_path.parent
        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels = source_labels(metrics["dataset"])
        source_map = manifest.get("source_train_indices")
        if source_map is None:
            raise RuntimeError(f"Missing source_train_indices: {manifest_path}")
        selected = []
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("selected", False):
                continue
            source_index = int(row.get("source_index", source_map[int(row["index"])]))
            true_label = int(labels[source_index])
            selected.append({
                "dataset": metrics["dataset"],
                "label_fraction": metrics["label_fraction"],
                "seed": metrics["seed"],
                "method": metrics["method"],
                "run_dir": str(run_dir),
                "source_index": source_index,
                "predicted_class": int(row["predicted_class"]),
                "true_class": true_label,
                "correct": int(int(row["predicted_class"]) == true_label),
                "calibrated_confidence": float(row["calibrated_confidence"]),
            })
        if selected:
            df = pd.DataFrame(selected)
            grouped = df.groupby(["dataset", "label_fraction", "seed", "method", "run_dir"], as_index=False)
            for keys, g in grouped:
                dataset, fraction, seed, method, run_dir_value = keys
                records.append({
                    "dataset": dataset,
                    "label_fraction": fraction,
                    "seed": seed,
                    "method": method,
                    "run_dir": run_dir_value,
                    "selected_count": len(g),
                    "correct_count": int(g["correct"].sum()),
                    "pseudo_label_precision": float(g["correct"].mean()),
                    "mean_confidence": float(g["calibrated_confidence"].mean()),
                    "class_counts_predicted": dict(Counter(g["predicted_class"])),
                    "class_precision": {
                        str(int(c)): float(x["correct"].mean())
                        for c, x in g.groupby("predicted_class")
                    },
                })
    out = pd.DataFrame(records)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} run-level pseudo-label audits to {OUT}")


if __name__ == "__main__":
    main()
