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
      "scholarships.com": {
        "title": ['h1'],
        "provider": ['[class*="provider"]','[class*="sponsor"]'],
        "amount": ['[class*="amount"]','[class*="award"]'],
        "deadline": ['[class*="deadline"]','time'],
        "award": ['[class*="award"]','[class*="benefit"]'],
      },
      "bigfuture.collegeboard.org": {
        "title": ['h1'],
        "provider": ['[class*="provider"]'],
        "amount": ['[class*="amount"]'],
        "deadline": ['[class*="deadline"]','time'],
        "award": ['[class*="award"]'],
      }
    }
    fallback = {
      "title": ['h1','h2'],
      "provider": ['[class*="provider"]','[class*="sponsor"]'],
      "amount": ['[class*="amount"]','[class*="award"]'],
      "deadline": ['[class*="deadline"]','time'],
      "award": ['[class*="award"]','[class*="benefit"]'],
    }
    s = sel.get(host, fallback)
    def g(k): return _text(_pick_first(soup, s.get(k, []))) or None
    return {k:g(k) for k in ["title","provider","amount","deadline","award"]}
