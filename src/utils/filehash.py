# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib

def sha1sum(data_or_path) -> str:
    h = hashlib.sha1()
    if isinstance(data_or_path, (bytes, bytearray)):
        h.update(data_or_path)
        return h.hexdigest()
    # assume file path
    with open(data_or_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
