# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Optional, Dict, Any

_CURRENCY_MAP = {
    "$": "USD", "usd": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
    "¥": "JPY", "jpy": "JPY",
    "₫": "VND", "vnd": "VND",
}

# Match: "$ 12,345.67", "USD 49.9", "€89", "VND 120000"
_PAT = re.compile(r"(?i)(usd|eur|gbp|jpy|vnd|\$|€|£|¥|₫)\s*([0-9][0-9\.,]*)")

def parse_money(text: str | None) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = str(text).strip()
    m = _PAT.search(s)
    if not m:
        return None
    sym, num = m.group(1), m.group(2)
    cur = _CURRENCY_MAP.get(sym.lower(), sym.upper())
    try:
        val = float(num.replace(",", ""))
    except Exception:
        return None
    return {"value": val, "currency": cur}
