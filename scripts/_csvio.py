"""Shared CSV reader for the results file (skips '#' comment lines)."""
import csv
from typing import Dict, List

FIELDS = ["kernel", "shape", "ours_gflops", "cublas_gflops", "pct_of_cublas",
          "tokens_per_s", "max_rel_err"]
NUMERIC = FIELDS[2:]


def load_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for rec in csv.DictReader(lines):
        for k in NUMERIC:
            rec[k] = float(rec[k])
        rows.append(rec)
    return rows


def is_illustrative(path: str) -> bool:
    """True if the file still carries the synthetic-data banner."""
    with open(path) as f:
        head = f.read(2048)
    return "ILLUSTRATIVE" in head or "SYNTHETIC" in head.upper()
