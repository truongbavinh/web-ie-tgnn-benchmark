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
      "zillow.com": {
        "title": ['h1', '[data-testid="home-details-summary-headline"]'],
        "location": ['[data-testid="home-details-summary-headline"]', '[class*="address"]'],
        "price": ['[data-testid="price"]', '[class*="price"]'],
        "area": ['[data-testid="living-area"]', '[class*="sqft"]', '[class*="m2"]'],
        "bedrooms": ['[data-testid="bed-bath-beyond"]', '[class*="bed"]'],
        "bathrooms": ['[data-testid="bed-bath-beyond"]', '[class*="bath"]'],
      },
      "rightmove.co.uk": {
        "title": ['h1', '.property-header-title'],
        "location": ['.property-header-title', '.primaryInfo'],
        "price": ['.property-header-price', '.prices'],
        "area": ['.floorplanSqFt', '.floorplanSqM'],
        "bedrooms": ['.bedrooms', '.key-features li:contains("bedroom")'],
        "bathrooms": ['.bathrooms', '.key-features li:contains("bathroom")'],
      }
    }
    fallback = {
        "title": ['h1','h2'],
        "location": ['[class*="address"]','[class*="location"]'],
        "price": ['[class*="price"]'],
        "area": ['[class*="area"]','[class*="sqft"]','[class*="m2"]'],
        "bedrooms": ['[class*="bed"]','li:contains("bed")'],
        "bathrooms": ['[class*="bath"]','li:contains("bath")'],
    }
    s = sel.get(host, fallback)
    def g(k): return _text(_pick_first(soup, s.get(k, []))) or None
    rec = {k:g(k) for k in ["title","location","price","area","bedrooms","bathrooms"]}
    # normalize simple numerics
    import re
    for k in ["bedrooms","bathrooms"]:
        v = rec[k]
        if isinstance(v,str):
            m = re.search(r'[0-9]+', v)
            rec[k] = int(m.group(0)) if m else None
    # area attempt
    v = rec["area"]
    if isinstance(v,str):
        m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(m2|sqm|sqft|ft2)', v, flags=re.I)
        if m:
            unit = m.group(2).lower().replace("ft2","sqft")
            rec["area"] = {"value": float(m.group(1)), "unit": unit}
    return rec
