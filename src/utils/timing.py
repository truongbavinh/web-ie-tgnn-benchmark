# -*- coding: utf-8 -*-
from __future__ import annotations
import time, contextlib

class Timer:
    def __init__(self, name: str = "", logger=None):
        self.name = name
        self.logger = logger
        self.start = None
        self.elapsed = 0.0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start
        if self.logger:
            self.logger.info(f"{self.name} took {self.elapsed:.3f}s")
        else:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")
