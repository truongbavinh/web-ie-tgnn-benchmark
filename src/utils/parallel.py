# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, Callable, Any, List, Optional
import multiprocessing as mp

def pmap(fn: Callable[[Any], Any], items: Iterable[Any], processes: Optional[int] = None, chunksize: int = 1) -> List[Any]:
    items = list(items)
    if not items:
        return []
    with mp.Pool(processes=processes or mp.cpu_count()) as pool:
        return list(pool.imap(fn, items, chunksize=chunksize))
