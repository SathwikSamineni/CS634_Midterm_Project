# src/brute_force.py
from __future__ import annotations
from itertools import combinations
from typing import List, Set, Dict, Tuple

def _support(itemset: Set[str], transactions: List[Set[str]]) -> float:
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions) if transactions else 0.0

def _all_itemsets(items: List[str], k: int):
    for comb in combinations(items, k):
        yield set(comb)

def frequent_itemsets_bruteforce(transactions: List[Set[str]], min_support: float) -> Dict[frozenset, float]:
    items = sorted({i for t in transactions for i in t})
    L: Dict[frozenset, float] = {}
    k = 1
    while True:
        any_freq = False
        for itemset in _all_itemsets(items, k):
            sup = _support(itemset, transactions)
            if sup >= min_support:
                L[frozenset(itemset)] = sup
                any_freq = True
        if not any_freq:
            break
        k += 1
    return L

def generate_rules(frequent_sets: Dict[frozenset, float], min_confidence: float):
    rules = []
    for L, supL in frequent_sets.items():
        if len(L) < 2: 
            continue
        L_items = set(L)
        for r in range(1, len(L)):
            for X_tuple in combinations(L, r):
                X = set(X_tuple)
                Y = L_items - X
                supX = frequent_sets.get(frozenset(X))
                if not supX:
                    continue
                conf = supL / supX
                if conf >= min_confidence:
                    rules.append((X, Y, supL, conf))
    rules.sort(key=lambda tup: (-tup[3], -tup[2]))
    return rules
