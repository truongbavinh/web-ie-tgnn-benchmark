#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from typing import List, Dict, Any
from metrics.slot_f1 import f1_slot_micro, f1_slot_macro
from metrics.exact_match_record import exact_match_record
from metrics.price_errors import price_mae, price_mape

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()
    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.pred)
    res = {}
    res.update(f1_slot_micro(gold, pred))
    res.update(f1_slot_macro(gold, pred))
    res.update(exact_match_record(gold, pred))
    res.update(price_mae(gold, pred))
    res.update(price_mape(gold, pred))
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
