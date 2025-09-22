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
import json

def extract_fields(html: str, url: Optional[str] = None) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    host = ""
    if url:
        import urllib.parse as up
        try: host = up.urlparse(url).netloc.lower()
        except: host = ""
    sel = {
      "booking.com": {
        "name": ['h2#hp_hotel_name', 'h1', '[data-testid="title"]'],
        "location": ['span.hp_address_subtitle', '[data-testid="address"]'],
        "price": ['.prco-valign-middle-helper', '[data-testid="price-and-discounted-price"]'],
        "rating": ['.b5cd09854e.d10a6220b4', '[data-testid="review-score"]'],
        "amenities": ['.hotel-facilities-group', 'ul[data-testid="property-highlights"] li'],
      },
      "hotels.com": {
        "name": ['h1', '[data-stid="content-hotel-title"]'],
        "location": ['[data-stid="content-hotel-lead-address"]', '.address'],
        "price": ['[data-stid="price-lockup-details"]', '.current-price'],
        "rating": ['[data-stid="content-hotel-reviews-rating"]', '.guest-reviews-badge__rating'],
        "amenities": ['.amenities', 'ul.amenity-list li'],
      },
      "kayak.com": {
        "name": ['h1', '[data-title]'],
        "location": ['.m-hotel-overview__sub-title', '.location'],
        "price": ['[data-hotel-price]', '.Common-Booking-MultiBookProvider__price'],
        "rating": ['[data-review-score]', '.rating-score'],
        "amenities": ['.amenities', 'ul li'],
      },
    }
    fallback = {
      "name": ['h1','h2','[itemprop="name"]'],
      "location": ['[class*="location"]','[itemprop="address"]'],
      "price": ['[class*="price"]','[data-price]'],
      "rating": ['[class*="rating"]','[data-rating]'],
      "amenities": ['[class*="amenit"] li','ul li'],
    }
    s = sel.get(host, fallback)
    name = _text(_pick_first(soup, s.get("name", []))) or None
    location = _text(_pick_first(soup, s.get("location", []))) or None
    price = _text(_pick_first(soup, s.get("price", []))) or None
    rating = _text(_pick_first(soup, s.get("rating", []))) or None
    ams = []
    for css in s.get("amenities", []):
        for el in soup.select(css):
            t=_text(el)
            if t: ams.append(t)
    # light price parse (no currency obj here; exporter may handle money)
    def parse_rating(r):
        try: return float(re.search(r'[0-9]+(?:\.[0-9]+)?', r).group(0))
        except: return None
    return {"name": name, "location": location, "price": price, "rating": parse_rating(rating), "amenities": list(dict.fromkeys(ams)) or None}
