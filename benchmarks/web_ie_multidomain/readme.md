# Web IE (Multi-domain) — Gold/Test Set

This folder contains the **evaluation set** for our multi-domain web information
extraction benchmark. Each record is a web page (or product/listing/event/app page),
annotated with **domain** and **attributes** to extract.

- Format: JSON Lines (`gold.jsonl`)
- ID key: `id` (unique per page)
- Domain key: `domain` (see `domains.csv`)
- Attribute container: `attributes` (domain-specific slots)
- Official schema: `schema.md`
- Domain inventory: `domains.csv`

> ⚠️ Train/val data are not included here. This directory is intended for **test/evaluation**
> by reviewers and for leaderboard reproduction.

## How predictions should look
Your model should produce a file `predictions.jsonl` with the **same `id`** and `domain`
values as the gold file, plus extracted `attributes`. Example below.

### Minimal run (example)
```bash
# Inference -> write predictions
python src/infer.py \
  --checkpoint checkpoints/gnn_bert.ckpt \
  --data benchmarks/web_ie_multidomain/gold.jsonl \
  --out results/ours/predictions.jsonl

# Evaluate
python scripts/evaluate.py \
  --gold benchmarks/web_ie_multidomain/gold.jsonl \
  --pred results/ours/predictions.jsonl \
  --task benchmarks/tasks.yaml \
  --out results/ours/summary.json
