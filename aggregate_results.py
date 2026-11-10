import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
rows = []
for path in sorted((ROOT / "results" / "runs").glob("*/metrics.json")):
    row = json.loads(path.read_text(encoding="utf-8"))
    row["run_dir"] = str(path.parent)
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No metrics.json files found")

scalar = [c for c in ["accuracy", "macro_f1", "weighted_f1", "ece", "brier_score", "nll", "pseudo_label_count", "temperature"] if c in df.columns]
df.to_csv(ROOT / "results" / "all_runs_audited.csv", index=False)
summary = df.groupby(["dataset", "label_fraction", "method"])[scalar].agg(["mean", "std", "count"]).reset_index()
summary.to_csv(ROOT / "results" / "summary_audited.csv", index=False)

main_methods = {"lora", "rc_ccb_lora", "dcr_ccb_lora"}
main = df[df.method.isin(main_methods)]
main.to_csv(ROOT / "results" / "mainline_runs.csv", index=False)

ablation_methods = {"rc_no_calibration", "dcr_no_calibration", "dcr_no_pseudo", "dcr_ccb_lora", "rc_ccb_lora", "lora"}
df[df.method.isin(ablation_methods)].to_csv(ROOT / "results" / "ablation_runs.csv", index=False)

report = []
report.append("# Local experiment audit\n")
report.append("This report is generated only from metrics.json files produced by the reconstructed local runner.\n")
report.append(f"- Runs found: {len(df)}")
report.append(f"- Datasets: {', '.join(sorted(df.dataset.unique()))}")
report.append(f"- Methods: {', '.join(sorted(df.method.unique()))}")
report.append("\n## Mainline summary\n")
report.append(summary[summary[('method', '')] if False else 'method'].to_string(index=False) if False else summary.to_string(index=False))
report.append("\n## Run-level records\n")
report.append(df[[c for c in ["dataset", "label_fraction", "seed", "method", "n_train", "n_labeled", "n_unlabeled", "temperature", "pseudo_label_count", "macro_f1", "ece", "pilot_caps", "run_dir"] if c in df.columns]].sort_values(["dataset", "seed", "method"]).to_string(index=False))
(ROOT / "results" / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
print(f"Wrote {len(df)} audited runs")
