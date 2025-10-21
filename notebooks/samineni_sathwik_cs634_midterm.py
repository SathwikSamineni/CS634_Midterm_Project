#!/usr/bin/env python
# coding: utf-8

# In[54]:


from pathlib import Path
import sys

def find_project_root(start: Path, must_have=("src","data"), max_up=10) -> Path:
    cur = start
    for _ in range(max_up):
        if all((cur / d).exists() for d in must_have):
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not find a folder with {must_have} starting from {start}")

PROJECT_ROOT = find_project_root(Path.cwd(), ("src","data"))
SRC_DIR  = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_OUT_DIR = PROJECT_ROOT / "notebooks" / "outputs"
NOTEBOOK_OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in sorted(DATA_DIR.glob("*_transactions.csv")):
    print("  -", p.name)


# In[55]:


import sys, os, time, csv
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

PROJECT_ROOT = Path.cwd()  
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and SRC_DIR.as_posix() not in sys.path:
    sys.path.append(SRC_DIR.as_posix())

from brute_force import frequent_itemsets_bruteforce, generate_rules


# In[56]:


DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATASET_INDEX = 5 
MIN_SUPPORT = 0.08
MIN_CONFIDENCE = 0.6

def list_txn_files(data_dir: Path):
    files = sorted(p for p in data_dir.glob("*_transactions.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No '*_transactions.csv' files in {data_dir}")
    def friendly(stem: str) -> str:
        s = stem.replace("_transactions","").replace("_"," ").strip().lower()
        s = (s.replace("sathwik","Sathwik")
              .replace("restaurant bar","Restaurant & Bar")
              .replace("pharmacy","Pharmacy")
              .replace("wholefoods","Wholefoods")
              .replace("apple","Apple"))
        return " ".join(w.capitalize() if w.islower() else w for w in s.split())
    return [(friendly(p.stem), p) for p in files]


def read_txns(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "Transaction" not in (rdr.fieldnames or []):
            raise ValueError(f"{path.name} missing 'Transaction' column.")
        for r in rdr:
            items = [x.strip() for x in (r.get("Transaction") or "").split(",") if x.strip()]
            if items: rows.append(set(items))
    if not rows:
        raise ValueError(f"No transactions in {path.name}.")
    return rows

def one_hot_bool(txns):
    items = sorted({i for t in txns for i in t})
    return pd.DataFrame([[i in t for i in items] for t in txns], columns=items, dtype=bool)

def fmt_set(s) -> str:
    return "{" + ", ".join(f"'{x}'" for x in sorted(map(str, s))) + "}"

def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else pd.DataFrame()).to_csv(path, index=False)


# In[62]:


DATA_DIR = Path(r"C:\Users\ASUS\Downloads\samineni_sathwik_midtermproject\data")

choices = list_txn_files(DATA_DIR)
print("Datasets found:")
for i, (name, path) in enumerate(choices, 1):
    print(f"  {i}. {name}  ({path.name})")


# In[ ]:


name, ds_path = choices[DATASET_INDEX - 1]
print(f"Using dataset: {name}  ({ds_path.name})")
print(f"min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}\n")

txns = read_txns(ds_path)
X = one_hot_bool(txns)

import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def fmt_set(s):
    return "{" + ", ".join(sorted(map(str, s))) + "}"

def keep_basic_cols(df):
    """Trim association_rules output to essential columns only."""
    return df[["antecedents", "consequents", "support", "confidence"]] if not df.empty else df

# ----- Brute Force -----
t0 = time.perf_counter()
L  = frequent_itemsets_bruteforce(txns, MIN_SUPPORT)
BR = generate_rules(L, MIN_CONFIDENCE)
t_b = time.perf_counter() - t0

Ldf = pd.DataFrame(
    [{"itemset": fmt_set(s), "support": sup} for s, sup in L.items()],
    columns=["itemset", "support"]
).sort_values("support", ascending=False, ignore_index=True)

BRdf = pd.DataFrame(
    [{"antecedent": fmt_set(a), "consequent": fmt_set(c), "support": s, "confidence": cfd}
     for a, c, s, cfd in BR],
    columns=["antecedent", "consequent", "support", "confidence"]
)
if not BRdf.empty:
    BRdf = BRdf.sort_values(["confidence", "support"], ascending=[False, False], ignore_index=True)

print("== Brute Force Frequent Itemsets ==")
display(Ldf.head(10))
print("== Brute Force Association Rule ==")
if BR:
    a, c, s, cf = sorted(BR, key=lambda r: (-r[3], -r[2]))[0]
    print(f"Rule 1: [{fmt_set(a)}, {fmt_set(c)}, {cf}]")
else:
    print("No association rules at these thresholds.")

# ----- Apriori -----
t0 = time.perf_counter()
FIa = apriori(X, min_support=MIN_SUPPORT, use_colnames=True)
RLa = association_rules(FIa, metric="confidence", min_threshold=MIN_CONFIDENCE) if not FIa.empty else pd.DataFrame()
RLa = keep_basic_cols(RLa)
t_a = time.perf_counter() - t0

print("\n== Apriori Frequent Itemsets ==")
display(FIa.head(10))
print("== Apriori Association Rule ==")
if not RLa.empty:
    r = RLa.sort_values(["confidence", "support"], ascending=[False, False]).iloc[0]
    print(f"Rule 1: [{fmt_set(r['antecedents'])}, {fmt_set(r['consequents'])}, {r['confidence']}]")
else:
    print("No association rules at these thresholds.")

# ----- FP-Growth -----
t0 = time.perf_counter()
FIf = fpgrowth(X, min_support=MIN_SUPPORT, use_colnames=True)
RLf = association_rules(FIf, metric="confidence", min_threshold=MIN_CONFIDENCE) if not FIf.empty else pd.DataFrame()
RLf = keep_basic_cols(RLf)
t_f = time.perf_counter() - t0

print("\n== FP-Growth Frequent Itemsets ==")
display(FIf.head(10))
print("== FP-Growth Association Rule ==")
if not RLf.empty:
    r = RLf.sort_values(["confidence", "support"], ascending=[False, False]).iloc[0]
    print(f"Rule 1: [{fmt_set(r['antecedents'])}, {fmt_set(r['consequents'])}, {r['confidence']}]")
else:
    print("No association rules at these thresholds.")

# ----- Timing Summary -----
timings = pd.DataFrame([
    {"Algorithm": "Brute Force", "Seconds": round(t_b, 6)},
    {"Algorithm": "Apriori",     "Seconds": round(t_a, 6)},
    {"Algorithm": "FP-Growth",   "Seconds": round(t_f, 6)},
])
print("\n== Timing Summary ==")
display(timings)

# Save to CSVs
out = OUTPUTS_DIR / ds_path.stem
save(Ldf, out/"bruteforce/frequent_itemsets.csv")
save(BRdf,out/"bruteforce/association_rules.csv")
save(FIa, out/"apriori/frequent_itemsets.csv")
save(RLa, out/"apriori/association_rules.csv")
save(FIf, out/"fpgrowth/frequent_itemsets.csv")
save(RLf, out/"fpgrowth/association_rules.csv")
print(f"\n Saved clean outputs under: {out}")

