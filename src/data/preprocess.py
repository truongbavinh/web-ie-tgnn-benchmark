# -*- coding: utf-8 -*-
"""Generic preprocessing utilities for multi-domain Web IE."""
from typing import Dict, List, Iterable, Optional, Tuple, Set
import os, json
from bs4 import BeautifulSoup

DEFAULT_IGNORE_TAGS: Set[str] = {"style", "script", "noscript", "meta", "link", "head"}

def _assign_node_ids(element, counter=None) -> None:
    if counter is None:
        counter = {'count': 0}
    element.node_id = str(counter['count'])
    counter['count'] += 1
    for child in element.children:
        if hasattr(child, 'children'):
            _assign_node_ids(child, counter)

def tokenize_and_label_html(html_path: str, ground_truth_for_file: Dict[str, Dict[str, str]],
                            ignore_tags: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
    ignore_tags = ignore_tags or DEFAULT_IGNORE_TAGS
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    _assign_node_ids(soup)
    lines: List[Tuple[str, str]] = []
    for tag in soup.find_all(True):
        if tag.name in ignore_tags:
            continue
        node_id = getattr(tag, "node_id", None)
        if node_id is None:
            continue
        raw_text = tag.get_text(" ", strip=True)
        if not raw_text:
            continue
        tokens = raw_text.split()
        if not tokens:
            continue
        gt = ground_truth_for_file.get(str(node_id))
        if gt:
            label_type = gt.get("label", "").strip()
            label_text = gt.get("text", "").strip()
            gt_tokens = set(label_text.split()) if label_text else set()
            matched = False
            for tok in tokens:
                if tok in gt_tokens and label_type:
                    prefix = "B-" if not matched else "I-"
                    lines.append((tok, f"{prefix}{label_type}"))
                    matched = True
                else:
                    lines.append((tok, "O"))
        else:
            for tok in tokens:
                lines.append((tok, "O"))
    return lines

def process_html_dir_to_bio(html_dir: str, ground_truth_json: str, out_dir: str,
                            ignore_tags: Optional[Set[str]] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(ground_truth_json, "r", encoding="utf-8") as f:
        gt_all = json.load(f)
    for fname in sorted(os.listdir(html_dir)):
        if not fname.endswith(".html"):
            continue
        fpath = os.path.join(html_dir, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{base}.bio")
        bio_lines = tokenize_and_label_html(fpath, gt_all.get(fname, {}), ignore_tags=ignore_tags)
        with open(out_path, "w", encoding="utf-8") as out:
            for tok, tag in bio_lines:
                out.write(f"{tok}\t{tag}\n")
        print(f"Wrote BIO: {out_path}")

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                import json
                yield json.loads(line)

def write_jsonl(path: str, records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
