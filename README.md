## 1) Folder Layout

```
samineni_sathwik_midtermproject/
│
├─ data/                                    # input datasets
│   ├─ apple_transactions.csv
│   ├─ wholefoods_transactions.csv
│   ├─ sathwik_pharmacy_transactions.csv
│   ├─ sathwik_hair_saloon_transactions.csv
│   └─ sathwik_restaurant_bar_transactions.csv
│
├─ notebooks/
│   ├─ cs634.ipynb                          # Part 6: notebook
│   ├─ cs634.py                             # auto-exported script version of ipynb
│   └─ outputs/                             # notebook-only outputs (kept separate)
│       └─ <dataset_stem>/
│           ├─ bruteforce/association_rules.csv
│           ├─ bruteforce/frequent_itemsets.csv
│           ├─ apriori/association_rules.csv
│           ├─ apriori/frequent_itemsets.csv
│           ├─ fpgrowth/association_rules.csv
│           └─ fpgrowth/frequent_itemsets.csv
│
├─ outputs/                                 # CLI/script outputs
│   └─ <dataset_stem>/
│       ├─ bruteforce/...
│       ├─ apriori/...
│       └─ fpgrowth/...
│
├─ src/
│   ├─ __init__.py                          # minimal to allow package-style imports
│   ├─ main.py                              # interactive CLI runner
│   ├─ brute_force.py                       # Brute Force: frequent sets + rule gen
│   ├─ apriori.py                           # Apriori wrapper 
│   └─ fp_growth.py                         # FP-Growth wrapper
│
├─ README.md                                # this file
└─ requirements.txt                         # pinned dependencies
```

> **Why two output locations?**
>
> * Running **scripts/CLI** saves to top-level `outputs/…` 
> * Running the **notebook** saves to `notebooks/outputs/…` so the two workflows never overwrite each other.

---

## 2) Datasets (input format)

All five CSVs live in `data/` and are deterministic (exactly **50 transactions** each).
The required schema is:

| Column           | Type   | Meaning                                                      |
| ---------------- | ------ | ------------------------------------------------------------ |
| `Transaction ID` | string | Identifier (e.g., `T001`). Not used in mining.               |
| `Transaction`    | string | **Comma-separated list of item names** (e.g., `Milk, Bread`) |

> The mining pipeline **parses `Transaction`** and converts each row into a set of items.

---

## 3) Environment Setup

Use **Python 3.9–3.12**.

```bash
# Windows PowerShell (recommended)
python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` (summary):

* `pandas`
* `mlxtend`
* `tabulate`
* `notebook` (if using Jupyter)

---

## 4) How to Run (CLI)

Run the main interactive script:

```bash
# from the project root
python src\main.py
```

You will be prompted in this order (defensive validation included):

1. **Dataset** – A numbered menu (1–5) pulled from `data/*_transactions.csv`
2. **Minimum support** – float in **(0, 1]**
3. **Minimum confidence** – float in **(0, 1]**

### Example session

```
Available datasets (from ../data):
  1. Apple
  2. Sathwik Hair Saloon
  3. Sathwik Pharmacy
  4. Sathwik Restaurant & Bar
  5. Wholefoods

Enter the number of the dataset to use (1–5): 2

Using dataset: Sathwik Hair Saloon  (sathwik_hair_saloon_transactions.csv)
Enter minimum support (0–1]: 0.05
Enter minimum confidence (0–1]: 0.5
```

Then the script:

* Runs **Brute Force**, **Apriori**, **FP-Growth**
* Prints **Frequent Itemsets** (top 20)
* Prints **exactly one association rule per algorithm** in the format:

```
Rule 1: [{'Item A'}, {'Item B'}, 0.5714285714285714]
```

* Saves CSV outputs under `outputs/<dataset_stem>/...`

### Where the CLI outputs go

```
outputs/<dataset_stem>/
├─ bruteforce/
│  ├─ frequent_itemsets.csv
│  └─ association_rules.csv
├─ apriori/
│  ├─ frequent_itemsets.csv
│  └─ association_rules.csv
└─ fpgrowth/
   ├─ frequent_itemsets.csv
   └─ association_rules.csv
```

---

## 5) How to Run (Jupyter Notebook)

Open **`notebooks/cs634.ipynb`** in Jupyter or VS Code.

### Notebook steps

1. **Run the first setup cell** (it auto-locates the project root and adds `src/` to `sys.path`).
2. **Show datasets**: run the “Datasets found” cell to see the numbered list.
3. Set **`DATASET_INDEX`, `MIN_SUPPORT`, `MIN_CONFIDENCE`** in the parameters cell.
4. Run the **execution** cell to:

   * Run Brute Force, Apriori, FP-Growth
   * Display top itemsets
   * Display one rule per algorithm (same format as CLI)
   * Show a timing summary
   * **Save CSV outputs under `notebooks/outputs/<dataset_stem>/…`**


### Export notebook to `.py` (required)

From Jupyter UI, or via CLI:

```bash
jupyter nbconvert --to script notebooks/cs634.ipynb
```

This creates `notebooks/cs634.py`.

---

## 6) What Each File Does

* **`src/main.py`**
  Interactive runner (Part 4). Lists datasets, validates inputs, runs all three algorithms, prints summarized outputs, and writes CSVs. Also prints exactly **one association rule** per algorithm in the specified “Rule 1: …” format.

* **`src/brute_force.py`**

  * `frequent_itemsets_bruteforce(transactions, min_support) -> Dict[frozenset,float]`
  * `generate_rules(frequent_sets, min_confidence) -> List[(A_set, C_set, support, confidence)]`
    Implements brute-force enumeration of all k-itemsets; then enumerates rules from frequent sets.

* **`src/apriori.py`** *(optional wrapper)*
  Thin wrapper around `mlxtend.frequent_patterns.apriori` + `association_rules` with sorting.

* **`src/fp_growth.py`** *(optional wrapper)*
  Thin wrapper around `mlxtend.frequent_patterns.fpgrowth` + `association_rules` with sorting.

* **`notebooks/cs634.ipynb`**
  Part 6: reproducible mining in Jupyter with clean outputs and separate notebook-only output folder.

* **`requirements.txt`**
  Dependencies for a clean environment.

* **`README.md`**
  This document.

---

## 7) Key Concepts & Evaluation (high-level)

* **Frequent Itemset Mining**: find sets of items with support ≥ threshold.
* **Association Rules**: implications `X → Y`, evaluated by:

  * **Support**: `support(X ∪ Y)`
  * **Confidence**: `support(X ∪ Y) / support(X)`

We keep only **support** and **confidence** in printed/saved rule outputs per rubric.
(Other metrics like lift/conviction are removed to keep outputs focused.)

* **Brute Force**: enumerates all k-itemsets until no frequent sets remain. Exponential in item count—baseline for correctness.
* **Apriori**: level-wise candidate generation using anti-monotonicity (prunes supersets of infrequent sets). Efficient for sparse data and moderate item counts.
* **FP-Growth**: compresses transactions into an FP-tree; mines without generating candidates explicitly. Efficient on dense datasets.

---

## 8) Defensive Programming & Validation

* Dataset menu shows only `*_transactions.csv` in `data/`.
* Must choose **exactly one** dataset (validated and re-prompted).
* `min_support`, `min_confidence` must be **floating-point in (0, 1]** (re-prompted).
* CSVs are validated to contain a **`Transaction`** column.
* One-hot encoding uses **boolean dtype** to avoid `mlxtend` warnings.
* Graceful messages if no frequent itemsets/rules at given thresholds; hints to adjust.

---

## 9) Tips & Troubleshooting

* **No frequent itemsets / no rules?**
  Lower thresholds (e.g., support `0.04–0.06`, confidence `0.5–0.6`) or increase repeated bundles in the dataset.

* **ModuleNotFoundError in notebook**
  Ensure the **first setup cell** adds `<project_root>/src` to `sys.path`. The provided notebook already does this with a robust root-finder.

* **Wrong outputs folder**

  * CLI saves to `outputs/…`
  * Notebook saves to `notebooks/outputs/…`
    This separation is intentional.

* **Windows path issues**
  Use raw string literals (e.g., `Path(r"C:\...\data")`) if you hard-set paths.

---

## 10) Example: Quick CLI Run

```powershell
# from project root
.venv\Scripts\activate
python src\main.py
# choose: 1..5
# enter minsup: 0.05
# enter minconf: 0.5

# outputs saved under:
# outputs/<dataset_stem>/{bruteforce,apriori,fpgrowth}/...
```

> Example printed rule line:

```
== Apriori Association Rule ==
Rule 1: [{'Waste Basket'}, {'Dish Rack'}, 0.5714285714285714]
```


