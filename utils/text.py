# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List

_SEP = re.compile(r"[,|/;•·]+")
_WS  = re.compile(r"\s+")

def smart_split(s: str | None) -> List[str]:
    """Tách chuỗi thành list theo dấu phẩy/gạch đứng/chấm phẩy/..."""
    if not s: return []
    parts = [p.strip() for p in _SEP.split(str(s)) if p.strip()]
    return parts

def normalize_ws(s: str | None) -> str:
    """Chuẩn hóa khoảng trắng (1 space)."""
    if not s: return ""
    return _WS.sub(" ", str(s)).strip()
