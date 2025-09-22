# -*- coding: utf-8 -*-
import logging, sys

def get_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """Logger chuẩn cho repo, không nhân đôi handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
    h.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    return logger
