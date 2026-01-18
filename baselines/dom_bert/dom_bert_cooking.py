#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOM+BERT for Cooking/Recipe IE (NAME, RATING, AUTHOR, TIME, TYPE)
- DOM-first regex backoff (recipe-oriented):
    * NAME: <title>, og:title, h1, JSON-LD name
    * RATING: JSON-LD aggregateRating/ratingValue, itemprop ratingValue, fallback "4.6/5"
    * AUTHOR: JSON-LD author, meta author, itemprop author, "By ..."
    * TIME: JSON-LD totalTime/cookTime/prepTime, time[itemprop], "Total: 45 mins"
    * TYPE: JSON-LD recipeCategory/recipeCuisine/recipeType, breadcrumbs, "Course: Dessert"
- Presence metric giống domain App (ALL5)
"""

import os, re, csv, json, glob, math, hashlib, html, pathlib, random, statistics
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup

from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DOMAIN         = "Cooking"
HTML_DIR       = "cooking_html"
MERGED_GOLD    = "cooking_gt_merged.json"
OUT_DIR        = "dombert_out"
BASE_MODEL     = "bert-base-uncased"

SEEDS          = [42, 43, 44]
TRAIN_RATIO    = 0.7
VAL_RATIO      = 0.1
MAX_EPOCHS     = 12
BATCH_SIZE     = 16
LR             = 5e-5
WARMUP_RATIO   = 0.1
MAX_TOKENS     = 512

SAVE_CKPT_DIR  = "dombert_ckpts"
USE_REGEX_BACKOFF = True

# Early Stopping
ES_MONITOR     = "presence"
ES_PATIENCE    = 3
ES_MIN_DELTA   = 0.002

# Label space (BIO)
LABELS = [
    "O",
    "B-NAME","I-NAME",
    "B-RATING","I-RATING",
    "B-AUTHOR","I-AUTHOR",
    "B-TIME","I-TIME",
    "B-TYPE","I-TYPE",
]
LABEL2ID = {lab:i for i,lab in enumerate(LABELS)}
ID2LABEL = {i:lab for lab,i in LABEL2ID.items()}

PRESENCE_LABELS = ["NAME","RATING","AUTHOR","TIME","TYPE"]

# =========================
# Utils
# =========================
def list_html_files(root: str) -> List[str]:
    pats = [os.path.join(root, "**", "*.html"), os.path.join(root, "**", "*.htm")]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files

def deterministic_split(paths: List[str], seed: int, train_ratio: float=0.7):
    scored = []
    for fp in paths:
        h = hashlib.md5((str(seed) + "|" + fp).encode("utf-8")).hexdigest()
        k = int(h[:8], 16)
        scored.append((k, fp))
    scored.sort(key=lambda x: x[0])
    n_train = int(math.floor(len(scored) * train_ratio))
    train = [fp for _, fp in scored[:n_train]]
    test  = [fp for _, fp in scored[n_train:]]
    return train, test

def deterministic_subsplit(paths: List[str], seed: int, val_ratio: float):
    scored = []
    for fp in paths:
        h = hashlib.md5((f"VAL|{seed}|{fp}").encode("utf-8")).hexdigest()
        k = int(h[:8], 16)
        scored.append((k, fp))
    scored.sort(key=lambda x: x[0])
    n_val = int(math.floor(len(scored) * val_ratio))
    val = [fp for _, fp in scored[:n_val]]
    train_sub = [fp for _, fp in scored[n_val:]]
    return train_sub, val

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def strip_html_to_text(raw_html: str) -> str:
    txt = re.sub(r"<!--.*?-->", " ", raw_html, flags=re.DOTALL)
    txt = re.sub(r"<script.*?>.*?</script>", " ", txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r"<style.*?>.*?</style>",  " ", txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_text(s: str) -> str:
    return normalize_space(s).lower()

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# =========================
# Gold loading
# =========================
MERGED_LABEL_MAP = {
    "name":"NAME",
    "rating":"RATING",
    "author":"AUTHOR",
    "time":"TIME",
    "type":"TYPE",
    # aliases phổ biến (tuỳ GT bạn)
    "recipe_name":"NAME",
    "recipe_title":"NAME",
    "creator":"AUTHOR",
    "by":"AUTHOR",
    "total_time":"TIME",
    "cook_time":"TIME",
    "prep_time":"TIME",
    "duration":"TIME",
    "category":"TYPE",
    "recipe_category":"TYPE",
    "recipe_cuisine":"TYPE",
    "course":"TYPE",
}

def load_merged_gold(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, Dict[str, List[str]]] = {}
    for html_name, items in (obj or {}).items():
        base = os.path.basename(html_name)
        bucket = defaultdict(list)
        for _id, rec in (items or {}).items():
            lab_raw = str(rec.get("label","")).strip().lower()
            txt_raw = (rec.get("text") or "").strip()
            if not lab_raw or not txt_raw:
                continue
            lab = MERGED_LABEL_MAP.get(lab_raw)
            if not lab:
                continue
            bucket[lab].append(normalize_space(txt_raw))
        if bucket:
            out[base] = dict(bucket)
    return out

# =========================
# Cooking normalizers + regex/DOM backoff
# =========================
RATING_NUM = re.compile(r"(?<!\d)([0-5](?:\.\d{1,2})?)(?!\d)")
RATING_SLASH5 = re.compile(r"\b([0-5](?:\.\d{1,2})?)\s*/\s*5\b", re.I)
RATING_STARS  = re.compile(r"\b([0-5](?:\.\d{1,2})?)\s*(?:stars?|★)\b", re.I)

AUTHOR_LABEL = re.compile(r"^\s*(author|by|written\s*by|recipe\s*by|contributor)\s*:?\s*$", re.I)
TIME_LABEL   = re.compile(r"^\s*(time|total\s*time|prep\s*time|cook\s*time|ready\s*in|duration)\s*:?\s*$", re.I)
TYPE_LABEL   = re.compile(r"^\s*(type|course|category|recipe\s*category|cuisine|recipe\s*cuisine|dish\s*type)\s*:?\s*$", re.I)

ISO8601_DUR = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", re.I)
TIME_TEXT = re.compile(r"\b(\d{1,3})\s*(seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|hr|h|days?|day|d)\b", re.I)

def normalize_rating(s: str) -> Optional[str]:
    t = (s or "").strip()
    if not t:
        return None
    m = RATING_SLASH5.search(t)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 5:
                return f"{v:.2f}".rstrip("0").rstrip(".")
        except:
            pass
    m = RATING_STARS.search(t)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 5:
                return f"{v:.2f}".rstrip("0").rstrip(".")
        except:
            pass
    nums = RATING_NUM.findall(t)
    for x in nums[:3]:
        try:
            v = float(x)
            if 0 <= v <= 5:
                return f"{v:.2f}".rstrip("0").rstrip(".")
        except:
            continue
    return None

def _strip_json_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//[^\n]*", r"\1", s)
    return s

def _rows_from_soup(soup: BeautifulSoup) -> List[Tuple[str,str]]:
    rows=[]
    for tr in soup.find_all("tr"):
        th = tr.find(["th","td"])
        tds = tr.find_all("td")
        if th and tds:
            lab = _clean(th.get_text(" ", strip=True))
            val = _clean(" ".join(td.get_text(" ", strip=True) for td in tds[1:]) or (tds[-1].get_text(" ", strip=True) if tds else ""))
            rows.append((lab, val))
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                rows.append((_clean(dt.get_text(" ", strip=True)), _clean(dd.get_text(" ", strip=True))))
    return rows

def _jsonld_find_first(soup: BeautifulSoup, keys: List[str]) -> Optional[str]:
    for sc in soup.find_all("script", attrs={"type":"application/ld+json"}):
        raw = (sc.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        def visit(obj):
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj:
                        v = obj[k]
                        if isinstance(v, (str,int,float)):
                            return str(v)
                        if isinstance(v, dict):
                            for kk in ["name","text","value","ratingValue"]:
                                if kk in v and isinstance(v[kk], (str,int,float)):
                                    return str(v[kk])
                            return json.dumps(v, ensure_ascii=False)
                        if isinstance(v, list):
                            return " ".join(str(x) for x in v[:5])
                for vv in obj.values():
                    r = visit(vv)
                    if r: return r
            elif isinstance(obj, list):
                for it in obj:
                    r = visit(it)
                    if r: return r
            return None

        r = visit(data)
        if r:
            return _clean(r)
    return None

def _parse_iso8601_duration(d: str) -> Optional[str]:
    """
    Convert ISO8601 duration (e.g., PT1H20M, PT45M, P1DT2H) to a readable short string.
    """
    if not d:
        return None
    d = d.strip()
    m = ISO8601_DUR.match(d)
    if not m:
        return None
    days = int(m.group(1) or 0)
    hrs  = int(m.group(2) or 0)
    mins = int(m.group(3) or 0)
    secs = int(m.group(4) or 0)
    parts = []
    if days: parts.append(f"{days} d")
    if hrs:  parts.append(f"{hrs} h")
    if mins: parts.append(f"{mins} min")
    if secs and not parts:  # chỉ show seconds khi không có d/h/m
        parts.append(f"{secs} s")
    return " ".join(parts) if parts else None

def normalize_time(s: str) -> Optional[str]:
    t = (s or "").strip()
    if not t:
        return None
    iso = _parse_iso8601_duration(t)
    if iso:
        return iso
    # gom các cụm "X h Y min" trong chuỗi
    hits = TIME_TEXT.findall(t)
    if hits:
        # giữ tối đa 3 thành phần theo thứ tự xuất hiện
        out = []
        for num, unit in hits[:3]:
            unit_l = unit.lower()
            if unit_l.startswith(("sec","s")):
                u = "s"
            elif unit_l.startswith(("min","m")):
                u = "min"
            elif unit_l.startswith(("hr","h","hour")):
                u = "h"
            elif unit_l.startswith(("day","d")):
                u = "d"
            else:
                u = unit_l
            out.append(f"{int(num)} {u}")
        return " ".join(out)
    return None

def bs4_find_name_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    og = soup.find("meta", attrs={"property":"og:title"})
    if og and og.get("content"):
        cands.append(_clean(og["content"]))

    if soup.title and soup.title.get_text(strip=True):
        cands.append(_clean(soup.title.get_text(" ", strip=True)))

    h1 = soup.find("h1")
    if h1:
        cands.append(_clean(h1.get_text(" ", strip=True)))

    jn = _jsonld_find_first(soup, ["name"])
    if jn:
        cands.append(_clean(jn))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_rating_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    # JSON-LD ratingValue / aggregateRating
    rv = _jsonld_find_first(soup, ["ratingValue"])
    if rv:
        nr = normalize_rating(rv)
        if nr: cands.append(nr)

    ar = _jsonld_find_first(soup, ["aggregateRating"])
    if ar:
        nr = normalize_rating(ar)
        if nr: cands.append(nr)

    m = soup.find(attrs={"itemprop":re.compile(r"ratingValue", re.I)})
    if m:
        txt = m.get("content") or m.get_text(" ", strip=True)
        nr = normalize_rating(txt)
        if nr: cands.append(nr)

    full = _clean(soup.get_text(" ", strip=True))
    for mm in re.finditer(r"\brating\b", full, flags=re.I):
        s = max(0, mm.start()-80); e = min(len(full), mm.end()+120)
        win = full[s:e]
        nr = normalize_rating(win)
        if nr:
            cands.append(nr)
            break

    out, seen = [], set()
    for v in cands:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_author_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    ja = _jsonld_find_first(soup, ["author"])
    if ja:
        cands.append(_clean(ja))

    ma = soup.find("meta", attrs={"name":re.compile(r"author", re.I)})
    if ma and ma.get("content"):
        cands.append(_clean(ma["content"]))

    # itemprop author (could be nested)
    ip = soup.find(attrs={"itemprop":re.compile(r"author", re.I)})
    if ip:
        txt = ip.get("content") or ip.get_text(" ", strip=True)
        if txt:
            cands.append(_clean(txt))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if AUTHOR_LABEL.match(lab) and val:
            cands.append(_clean(val))

    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\bBy\s+([^|•;\n]{2,80})", full, flags=re.I)
    if m:
        cands.append(_clean(m.group(1)))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v or len(v) > 120:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_time_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    # JSON-LD times (prefer totalTime)
    jt = _jsonld_find_first(soup, ["totalTime"])
    if jt:
        nt = normalize_time(jt) or jt
        cands.append(_clean(nt))

    # fallback cookTime/prepTime
    for k in ["cookTime","prepTime"]:
        jx = _jsonld_find_first(soup, [k])
        if jx:
            nt = normalize_time(jx) or jx
            cands.append(_clean(nt))

    # <time itemprop="totalTime|prepTime|cookTime">
    for ipk in ["totalTime","prepTime","cookTime"]:
        ttag = soup.find("time", attrs={"itemprop":re.compile(ipk, re.I)})
        if ttag:
            txt = ttag.get("datetime") or ttag.get_text(" ", strip=True)
            nt = normalize_time(txt) or txt
            if nt: cands.append(_clean(nt))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if TIME_LABEL.match(lab) and val:
            nt = normalize_time(val) or val
            cands.append(_clean(nt))

    # text scan "Total: 45 mins", "Ready in 1 hour"
    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Total\s*Time|Total|Ready\s*in|Cook\s*Time|Prep\s*Time)\s*[:\-–]\s*([^|•;\n]{2,80})",
                  full, flags=re.I)
    if m:
        nt = normalize_time(m.group(2)) or m.group(2)
        cands.append(_clean(nt))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_type_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    # JSON-LD keys for recipes
    jt = _jsonld_find_first(soup, ["recipeCategory","recipeCuisine","recipeType","category","genre"])
    if jt:
        cands.append(_clean(jt))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if TYPE_LABEL.match(lab) and val:
            cands.append(_clean(val))

    # breadcrumbs-ish
    for nav in soup.find_all(["nav","ol","ul"], attrs={"aria-label":re.compile(r"breadcrumb", re.I)}):
        txt = _clean(nav.get_text(" ", strip=True))
        if txt and len(txt) <= 160:
            parts = [p.strip() for p in re.split(r"[›>/|•]", txt) if p.strip()]
            if parts:
                cands.append(parts[-1])

    # inline "Course: Dessert" / "Cuisine: Italian"
    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Course|Category|Cuisine|Type)\s*[:\-–]\s*([^|•;\n]{2,80})", full, flags=re.I)
    if m:
        cands.append(_clean(m.group(2)))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def regex_backoff_candidates(text: str, raw_html: Optional[str]=None) -> Dict[str, List[str]]:
    """
    DOM-first (bs4_*), fallback regex from plain text.
    """
    res: Dict[str, List[str]] = {}

    if raw_html:
        n = bs4_find_name_candidates(raw_html)
        if n: res["NAME"] = n[:2]
        r = bs4_find_rating_candidates(raw_html)
        if r: res["RATING"] = r[:2]
        a = bs4_find_author_candidates(raw_html)
        if a: res["AUTHOR"] = a[:2]
        tm = bs4_find_time_candidates(raw_html)
        if tm: res["TIME"] = tm[:2]
        ty = bs4_find_type_candidates(raw_html)
        if ty: res["TYPE"] = ty[:2]

    t = text or ""

    if "NAME" not in res:
        head = t[:200]
        m = re.search(r"([A-Za-z0-9][A-Za-z0-9 \-\–\—\:\|•]{3,90})", head)
        if m:
            res["NAME"] = [_clean(m.group(1).split("|")[0])]

    if "RATING" not in res:
        m = RATING_SLASH5.search(t) or RATING_STARS.search(t)
        if m:
            nr = normalize_rating(m.group(0))
            if nr: res["RATING"] = [nr]
        else:
            mm = re.search(r"\brating\b.{0,40}([0-5](?:\.\d{1,2})?)", t, flags=re.I)
            if mm:
                nr = normalize_rating(mm.group(1))
                if nr: res["RATING"] = [nr]

    if "AUTHOR" not in res:
        m = re.search(r"\b(By|Author|Written\s*by)\s*[:\-–]?\s*([^|•;\n]{2,80})", t, flags=re.I)
        if m:
            res["AUTHOR"] = [_clean(m.group(2))]

    if "TIME" not in res:
        m = re.search(r"\b(Total\s*Time|Total|Ready\s*in|Cook\s*Time|Prep\s*Time)\s*[:\-–]\s*([^|•;\n]{2,80})",
                      t, flags=re.I)
        if m:
            nt = normalize_time(m.group(2)) or m.group(2)
            res["TIME"] = [_clean(nt)]
        else:
            nt = normalize_time(t[:500])
            if nt:
                res["TIME"] = [nt]

    if "TYPE" not in res:
        m = re.search(r"\b(Course|Category|Cuisine|Type)\s*[:\-–]\s*([^|•;\n]{2,80})", t, flags=re.I)
        if m:
            res["TYPE"] = [_clean(m.group(2))]

    return res

# =========================
# Dataset & Model (BIO)
# =========================
@dataclass
class Example:
    text: str
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]
    offset_mapping: List[Tuple[int,int]]
    labels: Optional[List[int]] = None
    file: str = ""

class DomBertDataset(Dataset):
    def __init__(self, files: List[str], tokenizer, split: str, gold_merged: Dict[str, Dict[str, List[str]]]):
        self.items: List[Example] = []
        for fp in tqdm(files, desc=f"Build {split}"):
            raw = read_file(fp)
            flat = strip_html_to_text(raw)
            if not flat:
                continue
            enc = tokenizer(flat, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=MAX_TOKENS)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            offs = enc["offset_mapping"]
            labels = [LABEL2ID["O"]] * len(input_ids)

            base = os.path.basename(fp)
            gold = gold_merged.get(base, {})

            def mark_spans(lst: List[str], ent: str):
                for gtxt in lst:
                    gnorm = normalize_text(gtxt)
                    if not gnorm:
                        continue
                    for m in re.finditer(re.escape(gnorm), normalize_text(flat)):
                        gs, ge = m.span()
                        started = False
                        for i, (s, e) in enumerate(offs):
                            if s == 0 and e == 0:
                                continue
                            if e <= gs or s >= ge:
                                continue
                            lab = f"I-{ent}"
                            if not started:
                                lab = f"B-{ent}"
                                started = True
                            labels[i] = LABEL2ID.get(lab, LABEL2ID["O"])

            for lab in PRESENCE_LABELS:
                vals = gold.get(lab, [])
                mark_spans(vals, lab)

            self.items.append(
                Example(
                    flat,
                    tokenizer.convert_ids_to_tokens(input_ids),
                    input_ids,
                    attn,
                    [(int(s), int(e)) for (s, e) in offs],
                    labels,
                    fp
                )
            )

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class DomBert(nn.Module):
    def __init__(self, base_model: str, num_labels: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.bert   = AutoModel.from_pretrained(base_model, config=self.config)
        H = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cls     = nn.Linear(H, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.cls(self.dropout(out))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

def collate(batch: List[Example]):
    input_ids = torch.tensor([ex.input_ids for ex in batch], dtype=torch.long)
    attn      = torch.tensor([ex.attention_mask for ex in batch], dtype=torch.long)
    labels    = torch.tensor([ex.labels for ex in batch], dtype=torch.long) if batch[0].labels is not None else None
    meta      = [{"file": ex.file, "offsets": ex.offset_mapping, "text": ex.text} for ex in batch]
    return input_ids, attn, labels, meta

# =========================
# Decode & Metrics
# =========================
def decode_bio_to_sets(meta, logits) -> Dict[str, List[str]]:
    text = meta["text"]
    offsets = meta["offsets"]
    pred_ids = logits.argmax(-1).tolist()

    spans = []
    cur_lab=None; cur_s=None; cur_e=None

    for pid, (s,e) in zip(pred_ids, offsets):
        if s == 0 and e == 0:
            if cur_lab is not None:
                spans.append((cur_lab, cur_s, cur_e))
                cur_lab=None
            continue

        lab = ID2LABEL.get(pid, "O")
        if lab == "O":
            if cur_lab is not None:
                spans.append((cur_lab, cur_s, cur_e))
                cur_lab=None
            continue

        if lab.startswith("B-"):
            if cur_lab is not None:
                spans.append((cur_lab, cur_s, cur_e))
            cur_lab = lab[2:]
            cur_s = s
            cur_e = e
        elif lab.startswith("I-"):
            l2 = lab[2:]
            if cur_lab == l2:
                cur_e = e
            else:
                cur_lab = l2
                cur_s = s
                cur_e = e

    if cur_lab is not None:
        spans.append((cur_lab, cur_s, cur_e))

    out = defaultdict(list)
    for lab, s, e in spans:
        val = normalize_space(text[s:e])

        if lab == "RATING":
            nv = normalize_rating(val)
            if nv: val = nv

        if lab == "TIME":
            nv = normalize_time(val)
            if nv: val = nv

        out[lab].append(val)

    return {k: sorted(set(v)) for k, v in out.items()}

def eval_presence_for_items(items: List[Example], perfile_preds: Dict[str, Dict[str, List[str]]], gold_merged: Dict[str, Dict[str, List[str]]]):
    n = 0
    sums = {k:0 for k in PRESENCE_LABELS}
    all5_sum = 0

    for ex in items:
        base = os.path.basename(ex.file)
        gold = gold_merged.get(base)
        if gold is None:
            continue
        preds = perfile_preds.get(ex.file, {})

        g = {lab: bool(gold.get(lab)) for lab in PRESENCE_LABELS}
        p = {lab: bool(preds.get(lab)) for lab in PRESENCE_LABELS}

        ok_all = True
        for lab in PRESENCE_LABELS:
            score = 100 if (g[lab] == p[lab]) else 0
            sums[lab] += score
            if score != 100:
                ok_all = False

        all5_sum += 100 if ok_all else 0
        n += 1

    if n == 0:
        return {lab:0.0 for lab in PRESENCE_LABELS} | {"ALL5":0.0}

    out = {lab: (sums[lab] / n) for lab in PRESENCE_LABELS}
    out["ALL5"] = all5_sum / n
    return out

# =========================
# Train / Infer
# =========================
def run_inference_collect_preds(model, loader, device) -> Dict[str, Dict[str, List[str]]]:
    perfile = {}
    model.eval()
    with torch.no_grad():
        for input_ids, attn, labels, meta in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attn, labels=None)
            dec = decode_bio_to_sets(meta[0], logits[0].cpu())

            if USE_REGEX_BACKOFF:
                raw = read_file(meta[0]["file"]) if os.path.isfile(meta[0]["file"]) else ""
                plain = strip_html_to_text(raw)
                rb = regex_backoff_candidates(plain, raw_html=raw)

                for lab in PRESENCE_LABELS:
                    if not dec.get(lab) and rb.get(lab):
                        dec[lab] = rb[lab]

            perfile[meta[0]["file"]] = dec
    return perfile

def train_one_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    files = list_html_files(HTML_DIR)
    if not files:
        raise RuntimeError(f"No HTML in {HTML_DIR}")

    train_files, test_files = deterministic_split(files, seed, TRAIN_RATIO)
    train_sub, val_files = deterministic_subsplit(train_files, seed, VAL_RATIO)

    gold_merged = load_merged_gold(MERGED_GOLD)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    ds_train = DomBertDataset(train_sub, tokenizer, "train", gold_merged)
    ds_val   = DomBertDataset(val_files,  tokenizer, "val",   gold_merged)
    ds_test  = DomBertDataset(test_files, tokenizer, "test",  gold_merged)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomBert(BASE_MODEL, num_labels=len(LABELS)).to(device)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(ds_test,  batch_size=1, shuffle=False, collate_fn=collate)

    if MAX_EPOCHS > 0:
        t_steps = max(1, len(train_loader) * MAX_EPOCHS)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sch = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * t_steps), t_steps)

        best_metric = None
        best_state = None
        patience = 0

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Seed {seed} Train ep{ep}")
            for input_ids, attn, labels, meta in pbar:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                _, loss = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                pbar.set_postfix(loss=float(loss))

            val_preds = run_inference_collect_preds(model, val_loader, device)
            pres = eval_presence_for_items(ds_val.items, val_preds, gold_merged)
            current = pres["ALL5"] if ES_MONITOR == "presence" else 0.0
            print(f"[VAL] ep{ep} presence " +
                  "/".join([f"{k}={pres[k]:.2f}" for k in PRESENCE_LABELS]) +
                  f" / ALL5={pres['ALL5']:.2f}")

            if best_metric is None or current > best_metric + ES_MIN_DELTA * 100.0:
                best_metric = current
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= ES_PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        ckpt_dir = os.path.join(SAVE_CKPT_DIR, f"{DOMAIN.lower()}_seed{seed}")
        pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    infer_and_eval_presence(seed, ds_test, model, test_loader, gold_merged)

def infer_and_eval_presence(seed: int, ds_test: DomBertDataset, model, loader, gold_merged):
    ensure_dir(OUT_DIR)
    device = next(model.parameters()).device

    pred_wide_csv = os.path.join(OUT_DIR, f"{DOMAIN.lower()}_seed{seed}_pred_wide.csv")
    met_csv       = os.path.join(OUT_DIR, f"{DOMAIN.lower()}_seed{seed}_metrics_presence.csv")

    perfile = {}

    with open(pred_wide_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["file","name","rating","author","time","type"])

        model.eval()
        with torch.no_grad():
            for input_ids, attn, labels, meta in tqdm(loader, desc=f"Seed {seed} Infer"):
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                logits, _ = model(input_ids=input_ids, attention_mask=attn, labels=None)
                dec = decode_bio_to_sets(meta[0], logits[0].cpu())

                if USE_REGEX_BACKOFF:
                    raw = read_file(meta[0]["file"]) if os.path.isfile(meta[0]["file"]) else ""
                    plain = strip_html_to_text(raw)
                    rb = regex_backoff_candidates(plain, raw_html=raw)
                    for lab in PRESENCE_LABELS:
                        if not dec.get(lab) and rb.get(lab):
                            dec[lab] = rb[lab]

                perfile[meta[0]["file"]] = dec

                def _one(lab: str) -> str:
                    vals = dec.get(lab, [])
                    if not vals:
                        return ""
                    v = vals[0]
                    if lab == "RATING":
                        return normalize_rating(v) or v
                    if lab == "TIME":
                        return normalize_time(v) or v
                    return v

                w.writerow([
                    meta[0]["file"],
                    _one("NAME"),
                    _one("RATING"),
                    _one("AUTHOR"),
                    _one("TIME"),
                    _one("TYPE"),
                ])

    print(f"[{DOMAIN}] Seed {seed} wide predictions → {pred_wide_csv}")

    totals = {"files":0, "all5_sum":0}
    sums = {lab:0 for lab in PRESENCE_LABELS}

    with open(met_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["seed","file"] + [f"{lab.lower()}_presence" for lab in PRESENCE_LABELS] + ["all5_presence"])

        for ex in ds_test.items:
            base = os.path.basename(ex.file)
            gold = gold_merged.get(base)
            if gold is None:
                continue
            preds = perfile.get(ex.file, {})

            g = {lab: bool(gold.get(lab)) for lab in PRESENCE_LABELS}
            p = {lab: bool(preds.get(lab)) for lab in PRESENCE_LABELS}

            row_scores = []
            ok_all = True
            for lab in PRESENCE_LABELS:
                sc = 100 if (g[lab] == p[lab]) else 0
                row_scores.append(sc)
                sums[lab] += sc
                if sc != 100:
                    ok_all = False

            all5 = 100 if ok_all else 0
            totals["files"] += 1
            totals["all5_sum"] += all5

            w.writerow([seed, ex.file] + row_scores + [all5])

        if totals["files"] > 0:
            summary = [f"{(sums[lab]/totals['files']):.2f}" for lab in PRESENCE_LABELS]
            w.writerow([seed, "__SUMMARY__"] + summary + [f"{(totals['all5_sum']/totals['files']):.2f}"])

    print(f"[{DOMAIN}] Seed {seed} presence metrics → {met_csv}")

def aggregate_presence_across_seeds(out_dir: str, seeds: List[int]):
    summaries = []
    for sd in seeds:
        p = os.path.join(out_dir, f"{DOMAIN.lower()}_seed{sd}_metrics_presence.csv")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                if row["file"] == "__SUMMARY__":
                    rec = {"seed": sd}
                    for lab in PRESENCE_LABELS:
                        rec[lab.lower()] = float(row[f"{lab.lower()}_presence"])
                    rec["all5"] = float(row["all5_presence"])
                    summaries.append(rec)

    if not summaries:
        print("No presence summaries to aggregate.")
        return

    sum_csv = os.path.join(out_dir, f"{DOMAIN.lower()}_presence_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["seed"] + [f"mean_{lab.lower()}_presence" for lab in PRESENCE_LABELS] + ["mean_all5_presence"])
        for r in sorted(summaries, key=lambda x: x["seed"]):
            w.writerow([r["seed"]] + [f"{r[lab.lower()]:.2f}" for lab in PRESENCE_LABELS] + [f"{r['all5']:.2f}"])

    agg_csv = os.path.join(out_dir, f"{DOMAIN.lower()}_presence_aggregate.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        cols = [lab.lower() for lab in PRESENCE_LABELS] + ["all5"]
        w.writerow(["domain","seeds"] +
                   [f"mean_{c}" for c in cols] +
                   [f"std_{c}" for c in cols] +
                   ["note"])

        means = {c: statistics.mean([r[c] for r in summaries]) for c in cols}
        stds  = {c: (statistics.pstdev([r[c] for r in summaries]) if len(summaries) > 1 else 0.0) for c in cols}

        w.writerow([
            DOMAIN,
            ",".join(str(r["seed"]) for r in sorted(summaries, key=lambda x: x["seed"])),
            *[f"{means[c]:.2f}" for c in cols],
            *[f"{stds[c]:.2f}" for c in cols],
            "Presence on test files with gold; NAME/RATING/AUTHOR/TIME/TYPE + ALL5"
        ])

    print(f"✓ Wrote per-seed presence summary: {sum_csv}")
    print(f"✓ Wrote presence aggregate:       {agg_csv}")

# =========================
# Main
# =========================
def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    files = list_html_files(HTML_DIR)
    if not files:
        raise RuntimeError(f"No HTML in {HTML_DIR}")
    for sd in SEEDS:
        train_one_seed(sd)
    aggregate_presence_across_seeds(OUT_DIR, SEEDS)

if __name__ == "__main__":
    main()
