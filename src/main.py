# src/main.py
from __future__ import annotations
import csv, sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from brute_force import frequent_itemsets_bruteforce, generate_rules
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
OUT_DIR  = (HERE / ".." / "outputs").resolve()

fmt_set = lambda s: "{" + ", ".join(f"'{x}'" for x in sorted(map(str, s))) + "}"
def friendly(stem:str)->str:
    s = stem.replace("_transactions","").replace("_"," ").strip().lower()
    s = (s.replace("restaurant bar","Restaurant & Bar")
           .replace("pharmacy","Pharmacy").replace("wholefoods","Wholefoods")
           .replace("apple","Apple"))
    return " ".join(w.capitalize() if w.islower() else w for w in s.split())

def list_txn_files():
    if not DATA_DIR.exists(): sys.exit(f"ERROR: data folder not found → {DATA_DIR}")
    files = sorted(DATA_DIR.glob("*_transactions.csv"))
    if not files:          sys.exit(f"ERROR: no '*_transactions.csv' in {DATA_DIR}")
    return [(friendly(p.stem), p) for p in files]

def choose_one(opts):
    print("\nAvailable datasets (from ../data):")
    for i,(n,_) in enumerate(opts,1): print(f"  {i}. {n}")
    while True:
        s = input(f"\nEnter the number of the dataset to use (1–{len(opts)}): ").strip()
        if s.isdigit() and 1 <= int(s) <= len(opts): return opts[int(s)-1]
        print("Invalid selection. Enter a single number from the list.")

prob = lambda label: next(v for _ in iter(int,1)    # simple validated prompt in (0,1]
                          if (lambda x: x if 0.0 < x <= 1.0 else print("Value must be >0 and ≤1.")
                             )( (v:= (lambda s: float(s) if s.replace('.','',1).isdigit() else (_:=print('Enter numeric like 0.05 or 0.6')) or -1)
                                  (input(f"Enter {label} (0–1]: ").strip())) ) )

def read_txns(path:Path):
    rows=[]
    with path.open(newline="",encoding="utf-8") as f:
        rdr=csv.DictReader(f)
        if not rdr.fieldnames or "Transaction" not in rdr.fieldnames:
            sys.exit(f"ERROR: '{path.name}' must contain a 'Transaction' column.")
        for r in rdr:
            items=[x.strip() for x in (r.get("Transaction") or "").split(",") if x.strip()]
            if items: rows.append(set(items))
    if not rows: sys.exit(f"ERROR: no transactions in {path.name}.")
    return rows

def one_hot_bool(txns):
    items=sorted({i for t in txns for i in t})
    df=pd.DataFrame([[i in t for i in items] for t in txns], columns=items, dtype=bool)
    if df.empty: sys.exit("ERROR: one-hot matrix is empty.")
    return df

def save(df:pd.DataFrame, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True); (df if df is not None else pd.DataFrame()).to_csv(path, index=False)

def print_one_rule_bruteforce(rules):
    if not rules: print("No association rules at these thresholds."); return
    A,C,sup,conf = sorted(rules, key=lambda r:(-r[3],-r[2]))[0]
    print(f"Rule 1: [{fmt_set(A)}, {fmt_set(C)}, {conf}]")

def print_one_rule_df(df:pd.DataFrame):
    if df is None or df.empty: print("No association rules at these thresholds."); return
    r=df.sort_values(["confidence","support"],ascending=[False,False]).iloc[0]
    print(f"Rule 1: [{fmt_set(r['antecedents'])}, {fmt_set(r['consequents'])}, {r['confidence']}]")

# -------- main --------
def main():
    choices = list_txn_files()
    name,path = choose_one(choices)
    print(f"\nUsing dataset: {name}  ({path.name})")
    minsup  = prob("minimum support")
    minconf = prob("minimum confidence")
    print(f"\nParameters → min_support = {minsup}, min_confidence = {minconf}")

    txns = read_txns(path)
    X    = one_hot_bool(txns)
    out  = OUT_DIR / path.stem

    # Brute Force
    L  = frequent_itemsets_bruteforce(txns, minsup)
    BR = generate_rules(L, minconf)
    Ldf = pd.DataFrame([{"itemset": fmt_set(s), "support": sup} for s,sup in L.items()]).sort_values("support", ascending=False)
    print("\n== Brute Force Frequent Itemsets ==")
    print(tabulate(Ldf.head(20), headers="keys", tablefmt="github", showindex=False) if not Ldf.empty else "(no frequent itemsets)")
    print("\n== Brute Force Association Rule =="); print_one_rule_bruteforce(BR)
    save(Ldf, out / "bruteforce" / "frequent_itemsets.csv")
    save(pd.DataFrame([{"antecedent": fmt_set(a), "consequent": fmt_set(c), "support": s, "confidence": cfd}
                       for a,c,s,cfd in BR]), out / "bruteforce" / "association_rules.csv")

    # Apriori
    FIa = apriori(X, min_support=minsup, use_colnames=True)
    RLa = association_rules(FIa, metric="confidence", min_threshold=minconf) if not FIa.empty else pd.DataFrame()
    print("\n== Apriori Frequent Itemsets ==")
    print(tabulate(FIa.head(20), headers="keys", tablefmt="github", showindex=False) if not FIa.empty else "(no frequent itemsets)")
    print("\n== Apriori Association Rule =="); print_one_rule_df(RLa)
    save(FIa, out / "apriori" / "frequent_itemsets.csv")
    save(RLa, out / "apriori" / "association_rules.csv")

    # FP-Growth
    FIf = fpgrowth(X, min_support=minsup, use_colnames=True)
    RLf = association_rules(FIf, metric="confidence", min_threshold=minconf) if not FIf.empty else pd.DataFrame()
    print("\n== FP-Growth Frequent Itemsets ==")
    print(tabulate(FIf.head(20), headers="keys", tablefmt="github", showindex=False) if not FIf.empty else "(no frequent itemsets)")
    print("\n== FP-Growth Association Rule =="); print_one_rule_df(RLf)
    save(FIf, out / "fpgrowth" / "frequent_itemsets.csv")
    save(RLf, out / "fpgrowth" / "association_rules.csv")

    if Ldf.empty and FIa.empty and FIf.empty:
        print("\n[Hint] No frequent itemsets at your support. Try a smaller support (e.g., 0.05).")
    elif (BR==[] and (RLa is None or RLa.empty) and (RLf is None or RLf.empty)):
        print("\n[Hint] No association rules at your thresholds. Lower support/confidence or strengthen repeated bundles.")

if __name__ == "__main__":
    main()
