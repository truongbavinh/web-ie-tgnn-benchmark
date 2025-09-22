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
      "coursera.org": {
        "title": ['h1','.banner-title'],
        "subject": ['.Breadcrumbs .breadcrumb-item', '[class*="subject"]'],
        "fees": ['[class*="price"]','[data-test="offer-price"]'],
        "duration": ['[class*="duration"]','[class*="hours"]'],
        "instructor": ['[class*="instructor"]', '.instructor-name'],
      },
      "edx.org": {
        "title": ['h1','.course-intro-heading'],
        "subject": ['.breadcrumb a','.course-subject'],
        "fees": ['[class*="price"]'],
        "duration": ['[class*="weeks"]','[class*="hours"]'],
        "instructor": ['[class*="instructor"]'],
      }
    }
    fallback = {
      "title": ['h1','h2'],
      "subject": ['[class*="subject"]','.breadcrumb a'],
      "fees": ['[class*="price"]'],
      "duration": ['[class*="duration"]','[class*="hours"]','[class*="weeks"]'],
      "instructor": ['[class*="instructor"]'],
    }
    s = sel.get(host, fallback)
    def g(k): return _text(_pick_first(soup, s.get(k, []))) or None
    return {k:g(k) for k in ["title","subject","fees","duration","instructor"]}
