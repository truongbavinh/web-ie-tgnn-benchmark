# -*- coding: utf-8 -*-
"""
domains.registry
— Dynamic domain registry loader for multi-domain Web IE.

Features:
- Auto-discover subpackages under `domains/` (fashion, hotel, ...)
- Load per-domain `config.yaml` (required/optional + site hints)
- Provide unified helpers:
    list_domains() -> List[str]
    get_domain_spec(domain) -> {"required": [...], "optional": [...]}
    preprocess(domain, html, url=None) -> Dict[str, Any]
    finalize(domain, attrs) -> Dict[str, Any]
    validate(domain, attrs, drop_unknown=True) -> Dict[str, Any]

Notes:
- If a domain lacks config.yaml, fallback specs are used (from schema.md).
- If a submodule lacks preprocessors/postprocess, the call becomes a no-op.

Typical usage:
    from domains.registry import preprocess, finalize, validate
    raw = preprocess("fashion", html, url)
    attrs = finalize("fashion", raw)
    attrs = validate("fashion", attrs)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import importlib
import pkgutil
from pathlib import Path

# Optional YAML; keep optional dependency
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # graceful fallback if not installed

# -------- Fallback spec (khớp schema.md) --------
FALLBACK_REQUIRED: Dict[str, List[str]] = {
    "tourist":     ["name","location","rating","price","duration"],
    "hotel":       ["name","location","price","rating","amenities"],
    "realestate":  ["title","location","price","area","bedrooms","bathrooms"],
    "flights":     ["name","duration","stops","price","departure_time","arrival_time","airline"],
    "fashion":     ["name","price"],
    "events":      ["name","venue","date_time","artists"],
    "app":         ["name","rating","category","developer","os"],
    "course":      ["title","subject","fees","duration","instructor"],
    "scholarships":["title","provider","amount","deadline","award"],
    "cooking":     ["name","rating","author","time","type"],
}
FALLBACK_OPTIONAL: Dict[str, List[str]] = {
    "fashion": ["material","color","size"],
    # others default to empty unless a config.yaml provides more
}

# -------- Discovery utilities --------
def _domains_pkg_path() -> Path:
    # Resolve filesystem path of this file → package root → domains/
    here = Path(__file__).resolve()
    return here.parent  # .../domains

def list_domains() -> List[str]:
    """List all domain subpackages (directories with __init__.py OR any module files)."""
    domains_dir = _domains_pkg_path()
    names: List[str] = []
    for m in pkgutil.iter_modules([str(domains_dir)]):
        # ignore this registry module itself
        if m.name in {"registry"}:
            continue
        # only keep packages or modules that are directories
        # Our domains are packages (dirs). Accept modules too for flexibility.
        names.append(m.name.lower())
    # Keep only known ones from schema if present, but allow extras
    return sorted(set(names) | set(FALLBACK_REQUIRED.keys()))

# -------- Config loading --------
_config_cache: Dict[str, Dict[str, Any]] = {}

def _load_config_yaml(domain: str) -> Optional[Dict[str, Any]]:
    """Load domains/<domain>/config.yaml if available."""
    if domain in _config_cache:
        return _config_cache[domain]
    cfg_path = _domains_pkg_path() / domain / "config.yaml"
    if cfg_path.exists() and yaml is not None:
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            _config_cache[domain] = cfg
            return cfg
        except Exception:
            # ignore parse error; fall back below
            pass
    _config_cache[domain] = {}
    return None

def get_domain_spec(domain: str) -> Dict[str, List[str]]:
    """Return {'required': [...], 'optional': [...]} for a domain."""
    d = domain.lower()
    cfg = _load_config_yaml(d) or {}
    req = list(cfg.get("required") or FALLBACK_REQUIRED.get(d, []))
    opt = list(cfg.get("optional") or FALLBACK_OPTIONAL.get(d, []))
    return {"required": req, "optional": opt}

# -------- Dynamic import helpers --------
_import_cache: Dict[Tuple[str, str], Any] = {}

def _maybe_import(domain: str, module_name: str):
    """
    Try import domains.<domain>.<module_name>.
    Cache results; return module or None.
    """
    key = (domain, module_name)
    if key in _import_cache:
        return _import_cache[key]
    mod = None
    fq = f"domains.{domain}.{module_name}"
    try:
        mod = importlib.import_module(fq)
    except Exception:
        mod = None
    _import_cache[key] = mod
    return mod

# -------- Public API: preprocess/finalize/validate --------
def preprocess(domain: str, html: str, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Call domains.<domain>.preprocessors.extract_fields(html, url) if available.
    Return {} if module or function missing.
    """
    d = domain.lower()
    mod = _maybe_import(d, "preprocessors")
    if mod and hasattr(mod, "extract_fields"):
        try:
            return mod.extract_fields(html, url=url)  # type: ignore[attr-defined]
        except Exception:
            # Fail safe: return empty dict so pipeline can continue
            return {}
    return {}

def finalize(domain: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call domains.<domain>.postprocess.finalize(attrs) if available, else return attrs.
    """
    d = domain.lower()
    mod = _maybe_import(d, "postprocess")
    if mod and hasattr(mod, "finalize"):
        try:
            return mod.finalize(attributes)  # type: ignore[attr-defined]
        except Exception:
            return dict(attributes)
    return dict(attributes)

def validate(domain: str, attributes: Dict[str, Any], drop_unknown: bool = True) -> Dict[str, Any]:
    """
    Ensure attributes conform to domain spec:
      - Drop empty/None/[]/{}
      - Optionally drop unknown keys not in required+optional
      - Does NOT enforce types; metrics/evaluators handle deep equality
    """
    spec = get_domain_spec(domain)
    allowed = set(spec["required"]) | set(spec["optional"])
    out: Dict[str, Any] = {}
    for k, v in attributes.items():
        if v in (None, "", [], {}):
            continue
        if drop_unknown and k not in allowed:
            continue
        out[k] = v
    return out

# -------- Convenience: end-to-end normalize --------
def normalize(domain: str, html: str, url: Optional[str] = None, drop_unknown: bool = True) -> Dict[str, Any]:
    """
    One-call pipeline commonly used for weak-labeling / rule baseline:
        html,url → extract_fields → finalize → validate
    """
    raw = preprocess(domain, html, url=url)
    fin = finalize(domain, raw)
    return validate(domain, fin, drop_unknown=drop_unknown)

# -------- Optional: currency map accessor (if provided in config.yaml) --------
def get_currency_map(domain: str) -> Dict[str, str]:
    """
    Return currency symbol/name -> ISO code map if present in config.yaml; else reasonable defaults.
    (Primarily used by fashion or other price-centric domains.)
    """
    cfg = _load_config_yaml(domain.lower()) or {}
    default = {
        "$": "USD", "usd": "USD",
        "€": "EUR", "eur": "EUR",
        "£": "GBP", "gbp": "GBP",
        "¥": "JPY", "jpy": "JPY",
        "₫": "VND", "vnd": "VND",
    }
    return dict(cfg.get("currency_map") or default)
