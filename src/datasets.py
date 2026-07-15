# src/datasets.py
"""Dataset loading helpers for SAST-IR experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_COUNTERFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "counterfact_50_strict.jsonl"

def _read_jsonl(path: Path) -> List[Dict]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            items.append(json.loads(line))
    return items

def load_counterfact_subset(n: int = 50, seed: int = 1337, path: Optional[str] = None) -> List[Dict]:
    """Load a deterministic sample from COUNTERFACT-Strict."""
    p = Path(path) if path else _DEFAULT_COUNTERFACT_PATH
    data = _read_jsonl(p)
    random.Random(seed).shuffle(data)
    if n and n > 0:
        data = data[:n]
    return data

def load_jsonl_dataset(path: str, n: Optional[int] = None, seed: int = 1337,
                       categories: Optional[List[str]] = None) -> List[Dict]:
    """Load JSONL records with optional case-insensitive category filtering."""
    data = _read_jsonl(Path(path))
    if categories:
        cats = {c.strip().lower() for c in categories if c and c.strip()}
        data = [ex for ex in data if str(ex.get("category","")).lower() in cats]
    random.Random(seed).shuffle(data)
    if n and n > 0:
        data = data[:n]
    return data
