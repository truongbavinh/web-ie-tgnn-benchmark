#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild a single unified `results/summary` by scanning all methods/domains/runs.

Usage:
  python scripts/rebuild_summary.py

This will:
  - Clean files inside `results/summary/` (not deleting the dir)
  - Scan: results/<method>/<domain>/run-*/summary.json and meta.json
  - Write: results/summary/by_run.csv
  - Write: results/summary/aggregate.csv + aggregate.json
"""
from __future__ import annotations
import json, csv, statistics
from pathlib import Path

RESULTS = Path("results")
SUMMARY = RESULTS / "summary"

def scan_runs():
    rows = []
    if not RESULTS.exists():
        return rows
    for method_dir in RESULTS.iterdir():
        if method_dir.name == "summary" or not method_dir.is_dir():
            continue
        method = method_dir.name
        for domain_dir in method_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for run_dir in domain_dir.glob("run-*"):
                if not run_dir.is_dir():
                    continue
                summary_path = run_dir / "summary.json"
                meta_path = run_dir / "meta.json"
                if not summary_path.exists():
                    continue
                try:
                    summ = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                run_id = run_dir.name.replace("run-","")
                row = {
                    "method": method,
                    "domain": domain,
                    "run_id": run_id,
                    "seed": meta.get("seed",""),
                    "exact_match_record": summ.get("exact_match_record",""),
                    "f1_slot_macro": summ.get("f1_slot_macro", summ.get("f1","")),
                    "f1_slot_micro": summ.get("f1_slot_micro", summ.get("f1","")),
                    "price_mae": summ.get("price_mae",""),
                    "price_mape": summ.get("price_mape",""),
                    "predictions_path": "",
                    "notes": meta.get("notes",""),
                }
                rows.append(row)
    return rows

def write_by_run(rows):
    SUMMARY.mkdir(parents=True, exist_ok=True)
    out = SUMMARY / "by_run.csv"
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    keys = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out

def write_aggregate(rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["domain"])].append(r)
    aggs = []
    for (m,d), rs in sorted(groups.items()):
        def tofloat(x):
            try: return float(x)
            except Exception: return None
        f1s = [tofloat(r["f1_slot_micro"]) for r in rs]
        f1s = [x for x in f1s if x is not None]
        if f1s:
            mean = statistics.mean(f1s)
            std  = statistics.pstdev(f1s) if len(f1s)>1 else 0.0
        else:
            mean = ""
            std  = ""
        aggs.append({
            "method": m,
            "domain": d,
            "f1_slot_micro_mean": mean,
            "f1_slot_micro_std": std,
            "f1_slot_macro_mean": mean,
            "f1_slot_macro_std": std,
            "n_runs": len(rs),
        })
    # write files
    csv_path = SUMMARY / "aggregate.csv"
    json_path = SUMMARY / "aggregate.json"
    if aggs:
        keys = list(aggs[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for a in aggs:
                w.writerow(a)
    json_path.write_text(json.dumps(aggs, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path

def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for p in list(SUMMARY.glob("*")):
        if p.is_file():
            p.unlink()
    rows = scan_runs()
    out_by_run = write_by_run(rows)
    out_csv, out_json = write_aggregate(rows)
    print(f"[OK] Rebuilt {out_by_run}")
    print(f"[OK] Rebuilt {out_csv} and {out_json}")

if __name__ == "__main__":
    main()
