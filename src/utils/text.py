# -*- coding: utf-8 -*-
"""Unicode normalization and small text utilities."""
from __future__ import annotations
import unicodedata, re

def nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip().lower()

def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def slugify(s: str, keep: str = "") -> str:
    s = unicodedata.normalize("NFKC", s or "").strip().lower()
    s = re.sub(rf"[^a-z0-9{re.escape(keep)}]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s
