# Results Overview

This folder contains **final evaluation artifacts** for all methods and domains in our multi-domain Web IE benchmark.  
Each method was run **3 times per domain** (one fixed-seed run + two randomized runs).

## Layout

```
results/
  ours/            # TGNN (our method)
  gcn_bert/        # Baseline: GCN + BERT
  llama/           # Baseline: Llama‑3.1‑8B‑Instruct
  mistral/         # Baseline: Mistral‑7B‑Instruct‑v0.2
  summary/
    by_run.csv
    aggregate.csv
    aggregate.json
```

Per‑method and per‑domain structure:

```
results/<method>/<domain>/
  run-seed42/
    predictions.jsonl    # optional; published when available
    summary.json         # metrics for this run
    meta.json            # seed, timestamp, commit, notes
  run-randA/
  run-randB/
```

**Domains:** `tourist`, `hotel`, `realestate`, `flights`, `fashion`, `events`, `app`, `course`, `scholarships`, `cooking`.

---

## What’s inside each file?

- **`summary.json`** – metrics for the run. When `predictions.jsonl` are not published, the values may come from the paper’s tables. Keys include:
  - `f1_slot_micro`, `f1_slot_macro` (F1 as reported or recomputed)
  - `precision`, `recall` (if available)
  - optional counters: `tp`, `fp`, `fn`, `pages`
- **`meta.json`** – metadata for reproducibility (seed, timestamp, commit, notes).
- **`predictions.jsonl`** (optional) – standardized per‑page predictions used for **independent verification** against gold labels.
- **`summary/by_run.csv`** – one row per run (method, domain, run_id, metrics).
- **`summary/aggregate.{csv,json}`** – mean/std metrics across the 3 runs for each `(method, domain)`.

> If you only need the final numbers, look at `summary/aggregate.csv`.
> If you want to audit each run, open the corresponding `summary.json` files.

---

## Verify results from published predictions

When `predictions.jsonl` are present for a run, you can recompute the metrics exactly as in the paper using the repo’s evaluation script:

```bash
python scripts/verify_results.py   --gold benchmarks/web_ie_multidomain/gold.jsonl   --pred results/<method>/<domain>/run-<seed42|randA|randB>/predictions.jsonl   --out results/<method>/<domain>/run-<...>/summary.json
```

This produces/overwrites `summary.json` for the run and is also suitable for CI checks.

> For large artifacts (graphs/checkpoints), please see `artifacts.yaml` and `scripts/download_artifacts.py` in the repo root. We do **not** commit multi‑GB files to git.

---

## Aggregate or update the summary tables

After adding new runs (or after verifying from predictions), regenerate the summary tables:

```bash
python scripts/aggregate_results.py
# → results/summary/aggregate.csv and aggregate.json
```

`aggregate.csv` contains mean and std over the 3 runs for each `(method, domain)`.

---

## Conventions

- **Run IDs:** `run-seed42`, `run-randA`, `run-randB`.
- **Methods:**
  - `ours` – TGNN (topological‑aware GNN + BERT)
  - `gcn_bert` – GCN + BERT baseline
  - `llama` – Llama‑3.1‑8B‑Instruct
  - `mistral` – Mistral‑7B‑Instruct‑v0.2
- **Schema:** prediction format is consistent with `benchmarks/web_ie_multidomain/schema.md`.

---

## Notes for reviewers

- We publish **metrics JSON** for all runs. When possible, we also include **predictions.jsonl** for auditability.
- If some predictions are hosted externally (due to size), their URLs and **SHA‑1** checksums are listed in `RESULTS.md` / `artifacts.yaml` at repo root.
- A minimal **sample** is included for a quick end‑to‑end sanity run (no large downloads).

If anything looks unclear, please see:
- `REPRODUCIBILITY.md` – proof‑of‑results workflow
- `RESULTS.md` – consolidated numbers and artifact checksums
