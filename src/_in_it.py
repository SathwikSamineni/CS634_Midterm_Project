# src/_in_it.py
from __future__ import annotations
from typing import List, Set

def load_transactions(csv_path: str) -> List[Set[str]]:
    txns: List[Set[str]] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        _ = next(f, None)  
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, items_str = line.split(",", 1)
            except ValueError:
                continue
            items = [x.strip() for x in items_str.split(",") if x.strip()]
            txns.append(set(items))
    return txns
def to_one_hot(transactions: List[Set[str]]):  
    import pandas as pd
    items = sorted({i for t in transactions for i in t})
    rows = [[1 if i in t else 0 for i in items] for t in transactions]
    return pd.DataFrame(rows, columns=items, dtype=int)
