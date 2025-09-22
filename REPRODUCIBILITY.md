# Reproducibility & Verification

This document explains how to verify the reported results and (optionally) reproduce runs.

## 1) Environment

- Python 3.10+
- Install PyTorch first (matching your CUDA/OS), then:
  ```bash
  pip install -r requirements.txt
  ```

## 2) Verify from Published Predictions

If predictions (`predictions.jsonl`) are available:

```bash
# Register a run (computes metrics from gold)
python scripts/register_run.py   --method <ours|gcn_bert|llama|mistral>   --domain <fashion|hotel|...>   --run_id <seed42|randA|randB>   --pred results/<method>/<domain>/run-<...>/predictions.jsonl   --gold benchmarks/web_ie_multidomain/gold.jsonl
```

Rebuild unified summary tables:

```bash
python scripts/rebuild_summary.py
# -> results/summary/by_run.csv, aggregate.csv, aggregate.json
```

## 3) Download Large Artifacts

Use `artifacts.yaml` and the helper script:

```bash
python scripts/download_artifacts.py --config artifacts.yaml --key <artifact_key>
python scripts/hash_file.py results/.../predictions.jsonl --algo sha1
```

## 4) Full Reproduction (Optional)

- Data preprocessing & graph building (`src/data/`)
- Train BERT/TGNN (`src/pipelines/` + `src/models/`)
- Export predictions (`src/exporters/`) → `predictions.jsonl`
- Register + aggregate as above

> Large intermediate files (graph tensors, checkpoints) are not committed; use `artifacts.yaml` instead.
