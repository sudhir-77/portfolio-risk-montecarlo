"""Reproduce every result: python run.py or python run.py --quick."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

from src.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
