#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, hashlib, sys, pathlib

def hash_file(path: pathlib.Path, algo: str="sha1") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="file path")
    ap.add_argument("--algo", default="sha1", choices=["sha1","sha256"])
    args = ap.parse_args()
    p = pathlib.Path(args.path)
    if not p.exists():
        print(f"[ERR] File not found: {p}", file=sys.stderr)
        sys.exit(1)
    print(hash_file(p, args.algo))

if __name__ == "__main__":
    main()
