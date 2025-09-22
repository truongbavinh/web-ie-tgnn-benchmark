# Sample Set (for CI sanity check only)

This folder contains a **tiny, synthetic classification sample** used only to keep
CI green. It is **not** part of the actual multi-domain attribute-extraction
benchmark. The real benchmark data (JSONL with domain + attributes) lives under
`benchmarks/all_domains/` (or your chosen folders) and follows `benchmarks/tasks.yaml`.

## Files
- `gold.csv` — ground-truth labels for a toy binary classification task.
  - Required columns: `id`, `label` (labels ∈ {0,1})
- (CI will generate) `results/examples/predictions.csv` — model predictions
  - Required columns: `id`, `prediction` (predictions ∈ {0,1})

## Why this exists
- Our default `scripts/evaluate.py` supports CSV (`id,label` vs `id,prediction`) out of the box,
  so we keep a tiny sample for a quick end-to-end check in GitHub Actions.
- The **real** evaluation for the paper uses JSONL attribute extraction and the
  metrics declared in `benchmarks/tasks.yaml` (e.g., `f1_slot_micro`, `exact_match_record`, etc.).
  Those live outside this `sample/` folder.

## Quick test
Create predictions (CI already does this using `baselines/baseline_random.py`):

```bash
python baselines/baseline_random.py \
  --gold benchmarks/sample/gold.csv \
  --out results/examples/predictions.csv
