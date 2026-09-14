"""PhDAI 732 Group 5 — PhiUSIIL leakage audit.

This file makes `src` a package. Without it the relative imports in the other
modules (`from . import config as C`) fail, and `python -m src.data` cannot
find anything. Do not delete it.
"""
