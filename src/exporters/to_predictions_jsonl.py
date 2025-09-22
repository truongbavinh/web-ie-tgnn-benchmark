# -*- coding: utf-8 -*-
"""CLI: Convert raw per-domain outputs to predictions.jsonl matching schema.md."""
import argparse, json
from exporters.postprocess import to_attributes

def main():
    ap = argparse.ArgumentParser(description="Convert raw model outputs (BIO or TGNN) to predictions.jsonl")
    ap.add_argument("--raw", required=True, help="Path to raw JSONL (one record per page)")
    ap.add_argument("--domain", required=True, help="Domain name (fashion/hotel/...) for fallback") 
    ap.add_argument("--out", required=True, help="Path to write predictions.jsonl")
    ap.add_argument("--label_map", required=False, help="Optional path to label_map JSON for TGNN node labels")
    args = ap.parse_args()

    lm = None
    if args.label_map:
        with open(args.label_map, "r", encoding="utf-8") as f:
            lm = json.load(f)

    to_attributes(args.raw, args.domain, args.out, label_map=lm)

if __name__ == "__main__":
    main()
