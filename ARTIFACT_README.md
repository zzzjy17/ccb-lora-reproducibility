# Release artifact

This directory is the GitHub release bundle for the paper experiment.

## Included

- Training, calibration, pseudo-label selection, audit, and aggregation code.
- The public dataset manifest with source URLs. Raw benchmark copies are intentionally omitted.
- The 50 formal protocol-v2 run directories for ChnSentiCorp and TNEWS.
- Run metrics, predictions, pseudo-label audit records, summary tables, and figure source data.
- The CSV source data used to build the paper figures.

Rendered PNG, PDF, and SVG figures are omitted because they are already part of the paper. Figure source CSV files are under `results/figure_data/`.

## Reproduce

1. Install `requirements.txt`.
2. Obtain the CLUE TNEWS files and pass their directory to `prepare_data.py`.
3. Let `prepare_data.py` download ChnSentiCorp from the listed Hugging Face source.
4. Run the command in `README.md` and then `aggregate_protocol_v2.py`.

The raw datasets remain under their original licenses and are not redistributed in this bundle. The run artifacts contain indices, labels used for evaluation, predictions, metrics, and selection decisions; they do not include the original text samples.
