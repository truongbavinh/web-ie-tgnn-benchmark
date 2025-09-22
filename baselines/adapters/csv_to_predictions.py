# -*- coding: utf-8 -*-
"""Convert baseline CSV (filename, name, price, material, color, size) to predictions.jsonl.

Usage:
  python baselines/adapters/csv_to_predictions.py \

    --csv baselines/llm/llm_output_fashion3.csv \

    --domain fashion \

    --out results/llm/predictions.jsonl
"""
from __future__ import annotations
import argparse, csv, json, os, re
from typing import Dict, Any, List

def parse_price_maybe(text: str):
    if not text: return None
    s = text.strip()
    # Try to split "<CUR> <NUM>" or detect symbols
    m = re.search(r"(?i)(usd|eur|gbp|jpy|vnd|\$|€|£|¥|₫)\s*([0-9][0-9\.,]*)", s)
    if m:
        sym, num = m.group(1), m.group(2)
        cmap = {"$":"USD","usd":"USD","€":"EUR","eur":"EUR","£":"GBP","gbp":"GBP","¥":"JPY","jpy":"JPY","vnd":"VND","₫":"VND"}
        cur = cmap.get(sym.lower(), sym.upper())
        try:
            val = float(num.replace(",",""))
            return {"value": val, "currency": cur}
        except Exception:
            return None
    # bare number -> unknown currency; leave as None to avoid schema mismatch
    return None

def to_attributes(row: Dict[str,str]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    if row.get("name"): attrs["name"] = row["name"].strip()
    pr = parse_price_maybe(row.get("price",""))
    if pr: attrs["price"] = pr
    if row.get("material"): 
        # split by comma/pipe
        parts = [p.strip() for p in re.split(r"[,|]+", row["material"]) if p.strip()]
        if parts: attrs["material"] = parts
    if row.get("color"): attrs["color"] = row["color"].strip()
    if row.get("size"): 
        parts = [p.strip() for p in re.split(r"[,|]+", row["size"]) if p.strip()]
        if parts: attrs["size"] = parts
    return attrs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--domain", required=True, help="e.g., fashion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.csv, newline="", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("id") or ""
            pid = os.path.splitext(os.path.basename(fname))[0]
            pred = {"id": pid, "domain": args.domain, "attributes": to_attributes(row)}
            out.write(json.dumps(pred, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
