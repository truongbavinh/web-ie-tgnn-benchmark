# Topology-Aware Web Information Extraction (Multi-Domain Benchmark)

This repository accompanies our paper on **topology-aware GNN + BERT (TGNN)** for multi-domain Web Information Extraction (IE).  
It contains: a standardized **benchmark**, **our TGNN method**, **baselines** (GCN+BERT, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2), and complete **reproducibility artifacts** (metrics and optional predictions).

> **At a glance**
> - 10 domains × 5 sites each (50 sites total)
> - Unified per-domain schemas & gold labels
> - Standard prediction format (`predictions.jsonl`)
> - 3 runs per method per domain (1 fixed seed + 2 randomized)

---

## Repository Layout

```
repo-root/
├─ benchmarks/
│  └─ web_ie_multidomain/
│     ├─ README.md
│     ├─ tasks.yaml
│     ├─ domains.csv                 # 10×5 site mapping
│     ├─ gold.jsonl                  # gold labels (standardized)
│     └─ schema.md                   # attribute schemas & examples
│
├─ src/
│  ├─ models/                        # BERT, GCN, TGNN definitions
│  ├─ data/                          # preprocess, samplers, graph builders
│  ├─ metrics/                       # slot-F1, exact-match, price metrics, etc.
│  ├─ exporters/                     # raw outputs → predictions.jsonl
│  └─ pipelines/                     # bert_pipeline.py, tgnn_pipeline.py
│
├─ baselines/
│  ├─ gcn_bert/                      # GCN+BERT wrappers & user scripts
│  ├─ llm/                           # Llama & Mistral extraction wrappers
│  └─ adapters/                      # CSV → predictions.jsonl converters
│
├─ results/
│  ├─ ours/                          # TGNN (seed42, randA, randB)
│  ├─ gcn_bert/                      # baseline 1
│  ├─ llama/                         # baseline 2
│  ├─ mistral/                       # baseline 3
│  └─ summary/                       # unified by_run.csv & aggregate.{csv,json}
│     └─ (see also results/README.md)
│
├─ utils/                            # small shared helpers (I/O, hashing, money, …)
├─ scripts/                          # register/verify/aggregate/rebuild_summary, etc.
├─ artifacts.yaml                    # URLs + hashes for large external artifacts
├─ tasks.yaml                        # benchmark task registry
└─ README.md                         # (this file)
```

---

## Domains & Schemas

We cover 10 domains with the following attributes:

| Domain        | Attributes (required unless noted)                                                   |
| ---           | ---                                                                                  |
| Tourist       | name, location, rating, price, duration                                              |
| Hotel         | name, location, price, rating, amenities                                             |
| RealEstate    | title, location, price, area, bedrooms, bathrooms                                    |
| Flights       | name, duration, stops, price, departure_time, arrival_time, airline                  |
| Fashion       | name, price, *(material, color, size optional)*                                      |
| Events        | name, venue, date_time, artists                                                      |
| App           | name, rating, category, developer, os                                                |
| Course        | title, subject, fees, duration, instructor                                           |
| Scholarships  | title, provider, amount, deadline, award                                             |
| Cooking       | name, rating, author, time, type                                                     |

See **`benchmarks/web_ie_multidomain/schema.md`** for strict types and examples (money fields, lists, datetimes, etc.).

---

## Environment & Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # install core deps
# If using CUDA/torch, install the torch build matching your driver
```

Optional deterministic seed in your scripts:
```python
from utils.seed import fix_seed
fix_seed(42)
```

---

## Large Artifacts (Graphs, Checkpoints, Big Predictions)

We do **not** commit multi-GB files. Instead, list them in **`artifacts.yaml`** with URLs and **SHA-1** checksums.

Download helper:
```bash
python scripts/download_artifacts.py --config artifacts.yaml --key <artifact_key>
```

Verify integrity:
```bash
python scripts/hash_file.py /path/to/file
```

---

## Standard Prediction Format

All methods produce `predictions.jsonl`, one JSON object per page:

```json
{"id":"page_0001","domain":"fashion","attributes":{
  "name":"Wool Jumper",
  "price":{"value":49.99,"currency":"USD"},
  "material":["wool"],
  "color":"navy",
  "size":["M","L"]
}}
```

- Monetary fields (`price`, `fees`, `amount`) → `{value, currency}` when parsable.
- List-like fields (`material`, `size`, `artists`, `amenities`) may be arrays.
- If your output is CSV, use **`baselines/adapters/csv_to_predictions.py`** (or `src/exporters/`) to convert to this format.

---

## Quick Verification (no training required)

If predictions are published (or downloaded via `artifacts.yaml`), you can recompute metrics:

```bash
# Register a run (computes metrics and logs it)
python scripts/register_run.py   --method ours   --domain fashion   --run_id seed42   --pred results/ours/fashion/run-seed42/predictions.jsonl   --gold benchmarks/web_ie_multidomain/gold.jsonl   --seed 42   --notes "TGNN fashion seeded"

# Aggregate mean/std across the 3 runs (seed42, randA, randB)
python scripts/aggregate_results.py
# → results/summary/aggregate.csv and aggregate.json
```

To rebuild a **single, unified** `results/summary/` by scanning the whole tree:
```bash
python scripts/rebuild_summary.py
```

This avoids duplicated summary folders and collates all methods/domains/runs into:
- `results/summary/by_run.csv`
- `results/summary/aggregate.{csv,json}`

---

## Running Baselines

### GCN+BERT
```bash
# (Optional) Download prebuilt graph tensors
python scripts/download_artifacts.py --config artifacts.yaml --key fashion_graph_pt

# Train / predict (wrappers call your original training code)
python baselines/gcn_bert/train.py
python baselines/gcn_bert/predict.py

# Convert CSV → predictions.jsonl
python baselines/adapters/csv_to_predictions.py   --csv baselines/gcn_bert/runs/fashion/gcn_predictions.csv   --domain fashion   --out results/gcn_bert/fashion/run-seed42/predictions.jsonl

# Register & aggregate (repeat for randA, randB)
python scripts/register_run.py --method gcn_bert --domain fashion --run_id seed42   --pred results/gcn_bert/fashion/run-seed42/predictions.jsonl   --gold benchmarks/web_ie_multidomain/gold.jsonl --seed 42
python scripts/aggregate_results.py
```

### LLMs (Llama-3.1-8B-Instruct / Mistral-7B-Instruct-v0.2)
```bash
# Produce CSV via your llm scripts
python baselines/llm/llama_extract.py
python baselines/llm/mistral_extract.py

# Convert, then register + aggregate as above
python baselines/adapters/csv_to_predictions.py   --csv baselines/llm/runs/fashion/llm_llama_fashion.csv   --domain fashion   --out results/llama/fashion/run-seed42/predictions.jsonl
python scripts/register_run.py --method llama --domain fashion --run_id seed42   --pred results/llama/fashion/run-seed42/predictions.jsonl   --gold benchmarks/web_ie_multidomain/gold.jsonl --seed 42
python scripts/aggregate_results.py
```

---

## Our Method (TGNN: Topology-Aware GNN + BERT)

Key components under `src/`:

- `src/models/` — TGNN, GCN, BERT models
- `src/pipelines/tgnn_pipeline.py` — end-to-end TGNN inference
- `src/data/` — HTML preprocess, PT graph builders, samplers
- `src/metrics/` — slot-F1 (micro/macro), exact-match, price MAE/MAPE
- `src/exporters/` — map raw outputs to standardized `predictions.jsonl`

Example (adapt paths to your environment/hardware):

```bash
python src/pipelines/tgnn_pipeline.py   --domain fashion   --input_dir data/fashion_html   --graph_dir baselines/gcn_bert/graph_pt_output   --out results/ours/fashion/run-seed42/predictions.jsonl   --seed 42
```

Then register and aggregate as shown above.

---

## Reproducibility

- **Three runs per domain:** `run-seed42`, `run-randA`, `run-randB`.
- **Per-run metrics:** `summary.json` stored alongside the run.
- **Unified summaries:** `results/summary/aggregate.csv` contains per-method, per-domain mean/std over the 3 runs.
- For large artifacts (graphs/checkpoints), use `artifacts.yaml` and keep **SHA-1** checksums.
- A minimal **sample** is provided so CI/e2e can run without large downloads.

---

## Citation

If you use this benchmark or code, please cite:

```bibtex
@inproceedings{yourkey2025topologywebie,
  title     = {Topology-Aware Graph Neural Networks + BERT for Multi-Domain Web Information Extraction},
  author    = {Your Name and Coauthors},
  booktitle = {Proceedings of ...},
  year      = {2025}
}
```

---

## License

Specify your license (e.g., **MIT**). For any third-party code/models, see their respective licenses.

---

## Contact

Questions or issues: open a GitHub Issue or email **youremail@example.com**.

---

### Changelog

- **v1.0** — Initial public release (benchmark, TGNN, baselines, reproducibility kit).
