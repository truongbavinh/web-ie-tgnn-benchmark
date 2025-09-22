#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download large artifacts listed in artifacts.yaml and verify SHA-1.

artifacts.yaml example:
artifacts:
  preds_ours_fashion:
    url: https://example.com/results/ours_fashion_seed42.jsonl
    sha1: 0123abcd...
    out: results/ours/fashion/run-seed42/predictions.jsonl
"""
import argparse, sys, os, pathlib, hashlib, yaml, requests

def sha1sum(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, out: pathlib.Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="artifacts.yaml")
    ap.add_argument("--key", required=True, help="artifact key in YAML")
    args = ap.parse_args()

    cfg_p = pathlib.Path(args.config)
    if not cfg_p.exists():
        print(f"[ERR] Config not found: {cfg_p}", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    art = data.get("artifacts", {}).get(args.key)
    if not art:
        print(f"[ERR] Key not found in artifacts: {args.key}", file=sys.stderr)
        sys.exit(1)

    url = art["url"]
    out = pathlib.Path(art["out"])
    expected_sha1 = art.get("sha1")
    print(f"[INFO] Downloading {args.key} from {url}")
    download(url, out)
    if expected_sha1:
        got = sha1sum(out)
        if got.lower() != expected_sha1.lower():
            print(f"[ERR] SHA1 mismatch! expected={expected_sha1} got={got}", file=sys.stderr)
            sys.exit(2)
        print(f"[OK] SHA1 verified: {got}")
    else:
        print("[WARN] No SHA1 provided. Skipping verification.")
    print(f"[OK] Saved to {out}")

if __name__ == "__main__":
    main()
