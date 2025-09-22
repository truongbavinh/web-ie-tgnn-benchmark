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
      "ticketmaster.com": {
        "name": ['h1', '[data-automation="event-details-title"]'],
        "venue": ['[data-automation="event-details-venue"]', '.venue-name'],
        "date_time": ['time', '[class*="date"]'],
        "artists": ['[class*="artist"]','[class*="headliner"]','ul li a[href*="artist"]'],
      },
      "songkick.com": {
        "name": ['h1', '.event-header h1'],
        "venue": ['.venue-name', 'a.venue-link'],
        "date_time": ['time', '.dateAndName time'],
        "artists": ['.line-up a', '.artists a'],
      }
    }
    fallback = {
      "name": ['h1','h2'],
      "venue": ['[class*="venue"]'],
      "date_time": ['time','[class*="date"]'],
      "artists": ['[class*="artist"] a','ul li a'],
    }
    s = sel.get(host, fallback)
    name = _text(_pick_first(soup, s.get("name", []))) or None
    venue = _text(_pick_first(soup, s.get("venue", []))) or None
    dt = _text(_pick_first(soup, s.get("date_time", []))) or None
    arts = []
    for css in s.get("artists", []):
        for el in soup.select(css):
            t=_text(el)
            if t: arts.append(t)
    return {"name":name,"venue":venue,"date_time":dt,"artists": list(dict.fromkeys(arts)) or None}
