# baselines/

This folder provides **reproducible baselines** and a **common interface** to produce `predictions.jsonl` compatible with `benchmarks/web_ie_multidomain/` and `scripts/evaluate_jsonl.py`.

Baselines included:
- **GCN+BERT** (graph over BERT features) — training & inference wrappers
- **LLM (Mistral-7B-Instruct v0.2)** — HTML → JSON via prompting
- **LLM (Llama‑3.1‑8B‑Instruct)** — HTML → JSON via prompting

Adapters are provided to convert each baseline's raw CSV/JSONL into the **unified** `predictions.jsonl` schema:
```json
{"id": "<page_id>", "domain": "<domain>", "attributes": {...}}
```

## Quick start

### 0) Environments
Install baseline dependencies (PyTorch/torch-geometric/Transformers etc.). Adjust CUDA/Torch versions as needed.
```bash
pip install -r baselines/requirements.txt
```

### 1) GCN+BERT (train → predict → adapt)
```bash
# Train (expects graph .pt files & class_weights.pt in ./graph_pt_output/)
python baselines/gcn_bert/train.py

# Predict to CSV
python baselines/gcn_bert/predict.py

# Convert CSV → predictions.jsonl (domain=fashion here)
python baselines/adapters/csv_to_predictions.py   --csv baselines/gcn_bert/gcn_predictions3.csv   --domain fashion   --out results/gcn_bert/predictions.jsonl
```

### 2) LLM — Mistral‑7B‑Instruct v0.2
```bash
# Run extractor over HTML list → CSV
python baselines/llm/mistral_extract.py

# Adapt CSV → predictions.jsonl
python baselines/adapters/csv_to_predictions.py   --csv baselines/llm/llm_output_fashion3.csv   --domain fashion   --out results/mistral/predictions.jsonl
```

### 3) LLM — Llama‑3.1‑8B‑Instruct
```bash
python baselines/llm/llama_extract.py

python baselines/adapters/csv_to_predictions.py   --csv baselines/llm/llm_llama_fashion3.csv   --domain fashion   --out results/llama/predictions.jsonl
```

### 4) Evaluate
```bash
python scripts/evaluate_jsonl.py   --gold benchmarks/web_ie_multidomain/gold.jsonl   --pred results/gcn_bert/predictions.jsonl
```

## Inputs expected

- **GCN+BERT**: graph `.pt` files under `graph_pt_output/` and `class_weights.pt` (see your training scripts). File names should correspond to page IDs.
- **LLM**: a CSV `file_list3.csv` with column `filename` listing HTML files in `fashion_html/`.

## Notes
- Both GCN+BERT training & inference in these wrappers **use self‑loop edges** at runtime as in your original code; topology from `.pt` is ignored on forward pass (see comments inside). 
- The adapters post‑process values lightly and leave money normalization to the metrics layer (schema‑conformant).
