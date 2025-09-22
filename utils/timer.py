# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from contextlib import contextmanager

@contextmanager
def Timer(label: str = ""):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if label:
            print(f"[Timer] {label}: {dt:.3f}s")
        else:
            print(f"[Timer] {dt:.3f}s")
