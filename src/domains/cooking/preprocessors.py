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
    host=""
    if url:
        import urllib.parse as up
        try: host = up.urlparse(url).netloc.lower()
        except: host = ""
    sel = {
      "allrecipes.com": {
        "name": ['h1','.article-heading'],
        "rating": ['[itemprop="ratingValue"]','[class*="rating"]'],
        "author": ['[itemprop="author"]','.author-name'],
        "time": ['[itemprop="totalTime"]','[class*="time"]'],
        "type": ['[class*="category"]','.breadcrumbs a'],
      },
      "foodnetwork.com": {
        "name": ['h1','.o-AssetTitle__a-HeadlineText'],
        "rating": ['[class*="Rating"]','[itemprop="ratingValue"]'],
        "author": ['[class*="author"]','[itemprop="author"]'],
        "time": ['[itemprop="totalTime"]','[class*="time"]'],
        "type": ['[class*="category"]','.breadcrumbs a'],
      }
    }
    fallback = {
      "name": ['h1','h2'],
      "rating": ['[class*="rating"]','[itemprop="ratingValue"]'],
      "author": ['[class*="author"]','[itemprop="author"]'],
      "time": ['[class*="time"]','[itemprop="totalTime"]'],
      "type": ['[class*="category"]','nav a'],
    }
    s = sel.get(host, fallback)
    def g(k): return _text(_pick_first(soup, s.get(k, []))) or None
    rec = {k:g(k) for k in ["name","rating","author","time","type"]}
    if isinstance(rec["rating"], str):
        import re
        m = re.search(r'[0-9]+(?:\.[0-9]+)?', rec["rating"])
        rec["rating"] = float(m.group(0)) if m else None
    return rec
