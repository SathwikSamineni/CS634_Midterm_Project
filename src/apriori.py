# src/apriori_lib.py
from __future__ import annotations
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(one_hot: pd.DataFrame, min_support: float, min_conf: float):
    fi = apriori(one_hot, min_support=min_support, use_colnames=True)
    rules = association_rules(fi, metric='confidence', min_threshold=min_conf)
    fi = fi.sort_values('support', ascending=False).reset_index(drop=True)
    rules = rules.sort_values(['confidence','support'], ascending=[False, False]).reset_index(drop=True)
    return fi, rules
