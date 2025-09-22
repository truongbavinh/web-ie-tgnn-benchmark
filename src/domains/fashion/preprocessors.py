# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional
import re, unicodedata
from bs4 import BeautifulSoup

_CURRENCY_MAP = {
    "$":"USD","usd":"USD","€":"EUR","eur":"EUR","£":"GBP","gbp":"GBP","¥":"JPY","jpy":"JPY","vnd":"VND","₫":"VND"
}
_NUM_RE = re.compile(r'(?<![A-Za-z])[0-9]+(?:[\.,][0-9]{1,2})?(?![A-Za-z])')

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

def parse_price(text: str) -> Optional[Dict[str, object]]:
    if not text: return None
    text = _nfkc(text)
    mcur = re.search(r'(?i)(usd|eur|gbp|jpy|vnd|\$|€|£|¥|₫)', text)
    cur = "USD"
    if mcur:
        sym = mcur.group(1).lower()
        cur = _CURRENCY_MAP.get(sym, sym.upper())
    mnum = _NUM_RE.search(text.replace(",", ""))
    if mnum:
        try:
            val = float(mnum.group(0))
            return {"value": val, "currency": cur}
        except Exception:
            return None
    return None

def extract_fields(html: str, url: Optional[str] = None) -> Dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    host = ""
    if url:
        import urllib.parse as up
        try:
            host = up.urlparse(url).netloc.lower()
        except Exception:
            host = ""

    site_map = {
        "zara.com": {
            "name": ['h1.product-name', 'h1', '[data-product-name]'],
            "price": ['span.price-current', 'span.sale', '[data-price]', '.price__amount', 'span[itemprop=price]'],
            "material": ['.composition', '.product-composition', 'li:contains("composition")', 'li:contains("Material")', 'li:contains("fabric")'],
            "color": ['.product-color', '.detail-color', 'span.color', '[data-color]'],
            "size": ['.size-selector button', '.product-sizes li', 'select#size option'],
        },
        "hm.com": {
            "name": ['h1', '.product-item-headline'],
            "price": ['.price', '.price-value', '[data-price]'],
            "material": ['.pdp-product-description-list-item:contains("Composition")', '.material'],
            "color": ['.color .value', '.primary-color'],
            "size": ['.picker-button', '.product-item-size .item'],
        },
        "farfetch.com": {
            "name": ['h1', '[data-component="ProductCardTitle"]'],
            "price": ['.price', '.priceValue'],
            "material": ['li.DetailListItem:contains("Composition")', 'li:contains("Material")'],
            "color": ['.Color', 'li:contains("Colour") ._value'],
            "size": ['.SizeOption', '.sizeSelector_button__*'],
        },
        "flannels.com": {
            "name": ['h1', '.productName'],
            "price": ['.pri', '.price', '.Price'],
            "material": ['li:contains("Fabric")', 'li:contains("Material")'],
            "color": ['.swatchValue', '.colourValue'],
            "size": ['.size .value', '.sizes button'],
        },
        "24s.com": {
            "name": ['h1', '.product-title'],
            "price": ['.product-price', '.price'],
            "material": ['.composition', '.product-details__composition'],
            "color": ['.product-color', '.color-label'],
            "size": ['.size-selector button', '.sizes__list button'],
        },
    }
    fallback = {
        "name": ['h1', 'h2', 'meta[property="og:title"]', 'meta[name="title"]'],
        "price": ['[class*="price"]', '[data-price]'],
        "material": ['li:contains("composition")', 'li:contains("material")', 'li:contains("fabric")'],
        "color": ['[class*="color"]', '[data-color]'],
        "size": ['[class*="size"] option', '[class*="size"] button', '[class*="size"] li'],
    }

    site_sel = site_map.get(host, fallback)

    name_el = _pick_first(soup, site_sel.get("name", []))
    price_el = _pick_first(soup, site_sel.get("price", []))
    material_el = _pick_first(soup, site_sel.get("material", []))
    color_el = _pick_first(soup, site_sel.get("color", []))
    sizes = []
    for css in site_sel.get("size", []):
        for el in soup.select(css):
            t = _text(el)
            if t: sizes.append(t)

    def dedup(seq):
        seen=set(); out=[]
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    name = _text(name_el)
    price_obj = parse_price(_text(price_el))
    materials = []
    if material_el:
        raw = _text(material_el)
        parts = re.split(r"[•·|,/;]+|\band\b|\+|\(|\)", raw, flags=re.I)
        materials = [p.strip() for p in parts if p.strip()]
    color = _text(color_el)
    sizes = dedup([s.strip() for s in sizes if s.strip()])

    return {
        "name": name or None,
        "price": price_obj,
        "material": materials or None,
        "color": color or None,
        "size": sizes or None,
    }
