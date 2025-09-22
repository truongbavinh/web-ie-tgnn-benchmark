# -*- coding: utf-8 -*-
import os, random
import numpy as np

def fix_seed(seed: int = 42):
    """Fix all random seeds (python, numpy, torch if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_torch_benchmark(enabled: bool = False):
    """Bật tắt cudnn.benchmark khi cần tốc độ (không deterministic)."""
    try:
        import torch
        torch.backends.cudnn.benchmark = bool(enabled)
    except Exception:
        pass
