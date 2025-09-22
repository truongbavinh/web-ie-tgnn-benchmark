# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional, Any
import re, unicodedata
from bs4 import BeautifulSoup

def _text(el) -> str:
    if not el: return ""
    return " ".join(el.get_text(" ", strip=True).split())

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _pick_first(soup, selectors: List[str]):
    for css in selectors:
        try:
            if ":contains(" in css:
                m = re.match(r'^(.*?)\:contains\("(.+?)"\)$', css)
                if not m: continue
                base, kw = m.group(1), m.group(2)
                for el in soup.select(base):
                    if kw.lower() in _text(el).lower():
                        return el
            else:
                el = soup.select_one(css)
                if el: return el
        except Exception:
            continue
    return None
def extract_fields(html: str, url: Optional[str]=None):
    soup = BeautifulSoup(html, "html.parser")
    host = ""
    if url:
        import urllib.parse as up
        try: host = up.urlparse(url).netloc.lower()
        except: host = ""
    sel = {
        "getyourguide.com": {
            "name": ['h1', '[data-test-id="activity-title"]'],
            "location": ['[data-test-id="activity-start-location"]', '[class*="location"]'],
            "rating": ['[data-test-id="rating-summary-value"]', '[class*="rating"]'],
            "price": ['[data-test-id="price-component"]', '[class*="price"]'],
            "duration": ['[data-test-id="duration"]', '[class*="duration"]'],
        },
        "klook.com": {
            "name": ['h1', '[class*="ProductTitle"]'],
            "location": ['[class*="Location"]'],
            "rating": ['[class*="Rating"]'],
            "price": ['[class*="Price"]'],
            "duration": ['[class*="Duration"]'],
        },
    }
    fallback = {
        "name": ['h1','h2'],
        "location": ['[class*="location"]'],
        "rating": ['[class*="rating"]'],
        "price": ['[class*="price"]'],
        "duration": ['[class*="duration"]','[class*="hour"]'],
    }
    s = sel.get(host, fallback)
    name = _text(_pick_first(soup, s.get("name", []))) or None
    location = _text(_pick_first(soup, s.get("location", []))) or None
    rating = _text(_pick_first(soup, s.get("rating", []))) or None
    price = _text(_pick_first(soup, s.get("price", []))) or None
    duration = _text(_pick_first(soup, s.get("duration", []))) or None
    try:
        import re
        r = float(re.search(r'[0-9]+(?:\.[0-9]+)?', rating).group(0)) if rating else None
    except: r = None
    return {"name":name,"location":location,"rating":r,"price":price,"duration":duration}
