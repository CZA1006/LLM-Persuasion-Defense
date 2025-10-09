# src/datasets.py
from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Optional

# 兼容原有函数：默认仍读旧的 counterfact_sample.jsonl
_DEF_CF_PATH = Path("data/counterfact_sample.jsonl")

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
    """保持向后兼容：默认读 data/counterfact_sample.jsonl"""
    p = Path(path) if path else _DEF_CF_PATH
    data = _read_jsonl(p)
    random.Random(seed).shuffle(data)
    if n and n > 0:
        data = data[:n]
    return data

def load_jsonl_dataset(path: str, n: Optional[int] = None, seed: int = 1337,
                       categories: Optional[List[str]] = None) -> List[Dict]:
    """
    通用加载器：支持按类别过滤 + 采样
    - path: JSONL 文件路径
    - categories: 仅保留 category 在该列表内的样本（大小写不敏感）
    """
    data = _read_jsonl(Path(path))
    if categories:
        cats = {c.strip().lower() for c in categories if c and c.strip()}
        data = [ex for ex in data if str(ex.get("category","")).lower() in cats]
    random.Random(seed).shuffle(data)
    if n and n > 0:
        data = data[:n]
    return data
