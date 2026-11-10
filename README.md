# Calibration-Aware Dynamic LoRA

Code and derived summaries for the paper **Calibration-Aware Dynamic Class-Balanced Pseudo-Labeling with LoRA for Low-Resource Chinese Text Classification**.

The main protocol uses `google-bert/bert-base-chinese`, a 10% labeled fraction, five seeds (`13`, `42`, `2026`, `7`, and `77`), and two datasets:

- ChnSentiCorp: official train, validation, and test splits.
- TNEWS: a stratified 12,000-example training cap and disjoint 5,000-example calibration and evaluation halves from the labeled development split.

The compared methods are ordinary LoRA, RC-CCB-LoRA, DCR-CCB-LoRA, full fine-tuning, and full fine-tuning with the same class-balanced pseudo-labeling stage. The DCR head adds a class-specific low-rank branch whose strength follows calibrated uncertainty.

## Setup

```text
python -m venv .venv
.venv\\Scripts\\activate       # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

The first run also downloads `google-bert/bert-base-chinese` through Transformers. GPU execution is recommended for the full protocol.

## Prepare data

Place the official CLUE `tnews_public` directory at `data/source/tnews_public`, or pass its location explicitly:

```text
python prepare_data.py --tnews-source path/to/tnews_public
```

ChnSentiCorp is downloaded from the Hugging Face dataset listed in `data/manifest.json`. IFLYTEK is not part of the paper's main protocol; prepare it only when needed:

```text
python prepare_data.py --include-iflytek --iflytek-source path/to/iflytek_public
```

The generated manifest contains repository-relative file names and public dataset sources. It does not record local source directories.

## Run experiments

The default command runs a small pilot:

```text
python run_experiments.py
```

For the protocol-v2 configuration, use the five seeds, the two datasets, and the method list used in the paper. The script writes run-level JSON files under `results/runs/` and aggregate CSV files under `results/`.

```text
python run_experiments.py --datasets ChnSentiCorp TNEWS --fractions 0.10 --seeds 13 42 2026 7 77 --methods lora rc_ccb_lora dcr_ccb_lora full_finetuning full_finetuning_ccb --run-tag protocol_v2
```

`aggregate_protocol_v2.py` checks the matched splits and summarizes the protocol-v2 runs. `audit_pseudolabel_truth_v2.py` performs the post-training source-label audit; those labels are not used during selection or retraining.

## Repository layout

```text
prepare_data.py                  dataset preparation
run_experiments.py               training, calibration, selection, evaluation
aggregate_protocol_v2.py         protocol validation and summaries
audit_pseudolabel_truth_v2.py    pseudo-label audit
figures/                         figure generators
data/                            prepared benchmark copies and manifest
results/                         derived summaries and optional run artifacts
```

The `results/runs/` directory contains large, historical run artifacts. It is useful for reproducing the reported tables but is not required to run the code. Do not commit model checkpoints, Python caches, local manuscript files, or private source paths.

The release package keeps only the 50 formal protocol-v2 run directories. Earlier pilot, ablation, IFLYTEK, and sensitivity runs stay out of the release.

Rendered figures are omitted from the release because they are already included in the paper. The tabular source data used for the figures stays under `results/figure_data/`.

## Data and licensing

The benchmarks remain subject to their original licenses and terms. Download them from the CLUE repository and Hugging Face source listed above. The repository contains prepared copies only when the project owner chooses to redistribute them.
