#!/usr/bin/env python3
"""
dataprep_final.py
Full, robust data-prep pipeline for the MXT quiz CSV:
- safe merge (handles list-like cells)
- column cleaning
- automatic quiz column detection
- auto-detect branching logic (categorical + numeric)
- apply multi-layer branching (mark Skipped)
- one-hot encode quiz responses
- Laplace smoothing (vectorized)
- numeric-only variance detection
- dependency tree visualization
- data dictionary output
"""

import os
import re
import ast
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# Paths / Config
# ----------------------------
RAW_CSV = "data/raw/quiz_product_table.csv"
PROCESSED_CSV = "data/processed/quiz_product_table_cleaned.csv"
DATA_DICT_CSV = "data/processed/data_dictionary.csv"
BRANCH_TREE_PNG = "data/processed/quiz_branching_tree.png"

os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
os.makedirs(os.path.dirname(DATA_DICT_CSV), exist_ok=True)
from pyairtable import Table

# Airtable config
from pyairtable import Api
import pandas as pd
import os

# ----------------------------
# Config
# ----------------------------
PAT = os.environ.get("AIRTABLE_PAT")  # <- read from environment variable
BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
TABLE_NAME = "Quiz and Product Table - Sandbox"
OUTPUT_PATH = "data/raw/quiz_product_table.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ----------------------------
# Connect to Airtable
# ----------------------------
api = Api(PAT)
table = api.table(BASE_ID, TABLE_NAME)

# Pull all records with pagination
all_records = []
offset = None

while True:
    response = table.all(page_size=100, offset=offset)
    if not response:
        break
    all_records.extend(response)
    offset = response[-1]['id'] if len(response) == 100 else None
    if not offset:
        break

# Convert to DataFrame
df = pd.DataFrame([r["fields"] for r in all_records])

# Save CSV
df.to_csv(OUTPUT_PATH, index=False)
print(f"Pulled {len(df)} rows successfully! Saved to {OUTPUT_PATH}")
print("Preview of first 5 rows:")
print(df.head())

# ----------------------------
# Helpers
# ----------------------------
def safe_parse_cell(x):
    """If cell is a Python-list string like "['a','b']" or actually a list, convert to tuple.
       Else return original."""
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, tuple):
        return x
    if isinstance(x, str):
        s = x.strip()
        # quick heuristic: string representations of lists start with [ and end with ]
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return tuple(parsed)
            except Exception:
                return x
    return x

def clean_colname(name):
    name = name.strip()
    # replace newlines, slashes, percent signs, multiple spaces
    name = re.sub(r'\s+', ' ', name)
    name = name.replace('\n', ' ').replace('/', '_').replace('%', 'pct')
    name = name.replace('?', '').replace('(', '').replace(')', '')
    # safe python identifier-ish
    name = re.sub(r'[^\w\s\-_]', '', name)
    name = name.strip().replace(' ', '_')
    return name

def is_probable_personal_col(col):
    col_l = col.lower()
    checks = ['name', 'email', 'phone', 'address', 'zip', 'postcode', 'city', 'state', 'country']
    return any(k in col_l for k in checks)

def is_product_col(col):
    col_l = col.lower()
    checks = ['sku', 'product', 'shopify', 'mxt_sku', 'mxt_sku_list']
    return any(k in col_l for k in checks)

def likely_quiz_col(col):
    # Heuristic: many columns here are question texts and contain words like 'what', 'how', 'which', 'select', '?'
    col_l = col.lower()
    triggers = ['select', 'what', 'how', 'which', 'do', 'are', 'when', 'response', 'concern', 'question', 'breakout', 'skin']
    # exclude explicit personal/product columns
    if is_probable_personal_col(col) or is_product_col(col):
        return False
    return any(t in col_l for t in triggers)

# ----------------------------
# Load CSV
# ----------------------------
print("Loading CSV:", RAW_CSV)
df = pd.read_csv(RAW_CSV, dtype=object)  # load everything as object to be safe
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

# ----------------------------
# Clean column names
# ----------------------------
orig_columns = df.columns.tolist()
cleaned = [clean_colname(c) for c in orig_columns]
rename_map = dict(zip(orig_columns, cleaned))
df.rename(columns=rename_map, inplace=True)
print("Columns cleaned. Example columns:", df.columns.tolist()[:12])

# ----------------------------
# Normalize list-like cells (stringified lists from Airtable)
# ----------------------------
for col in df.columns:
    # apply safe parsing on this column
    df[col] = df[col].map(safe_parse_cell)

# ----------------------------
# Deduplicate (if any list-like entries remain as lists, they've been converted to tuples)
# ----------------------------
# If any columns still contain unhashable types, replace them with string repr to ensure dedupe works.
def make_hashable_series(s):
    if s.apply(lambda x: isinstance(x, (list, dict))).any():
        return s.map(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x)
    return s

for col in df.columns:
    try:
        # attempt to factorize to detect unhashable types
        pd.factorize(df[col])
    except Exception:
        df[col] = make_hashable_series(df[col])

# Now drop duplicates safely
before = len(df)
df = df.drop_duplicates(ignore_index=True)
print(f"Dropped duplicates: before={before}, after={len(df)}")

# ----------------------------
# Fill missing values (categorical -> 'Unknown', numeric detection handled later)
# ----------------------------
df = df.fillna('Unknown')

# ----------------------------
# Auto-detect quiz response columns
# ----------------------------
# Use heuristics: likely_quiz_col OR contains '?' or 'response' or is long question text
candidate_cols = [c for c in df.columns if likely_quiz_col(c)]
# Also broaden with presence of 'response' or '?' originally
candidate_cols += [c for c in df.columns if ('response' in c.lower() or '?' in c)]
# unique and preserve order
quiz_cols = []
for c in candidate_cols:
    if c not in quiz_cols and c in df.columns:
        quiz_cols.append(c)

# If heuristics found none, fallback to any column with moderate unique values and object dtype
if len(quiz_cols) == 0:
    for c in df.columns:
        nunique = df[c].nunique()
        if 2 <= nunique <= max(50, int(0.02 * len(df))):
            if not is_probable_personal_col(c) and not is_product_col(c):
                quiz_cols.append(c)

print(f"Auto-detected {len(quiz_cols)} quiz response columns (preview): {quiz_cols[:10]}")

# ----------------------------
# Auto-detect branching logic (categorical + numeric)
# Approach:
# - For each quiz column parent_col, for each unique value v:
#   - check dependent columns: if among rows where parent==v, some dep_col has a large increase in 'Unknown'/'Skipped'/empty
# - For numeric detection: if parent looks numeric (all values numeric-like), find dependent cols where dep missing correlates with parent low/high
# ----------------------------
def is_numeric_series(s):
    # treat if at least 90% convertible to float
    try:
        conv = pd.to_numeric(s.replace('Unknown', np.nan), errors='coerce')
        pct_numeric = conv.notna().sum() / max(1, len(conv))
        return pct_numeric >= 0.9
    except Exception:
        return False

branch_logic = {}  # {parent_col: {trigger_value_or_condition: [child_cols]}}

for parent in quiz_cols:
    parent_series = df[parent].astype(str)
    unique_vals = parent_series.dropna().unique()
    # CATEGORICAL detection:
    for val in unique_vals:
        mask_parent = parent_series.str.lower() == str(val).lower()
        skipped_children = []
        for child in quiz_cols:
            if child == parent:
                continue
            # baseline unknown rate
            baseline_unknown = (df[child].astype(str).str.lower() == 'unknown').mean()
            # unknown rate when parent==val
            if mask_parent.sum() == 0:
                continue
            child_unknown_when = (df.loc[mask_parent, child].astype(str).str.lower() == 'unknown').mean()
            # if unknown rate for child increases significantly (heuristic: 50% relative increase and at least 0.1 absolute)
            if (child_unknown_when - baseline_unknown) > 0.10 and child_unknown_when > max(0.15, baseline_unknown * 1.5):
                skipped_children.append(child)
        if skipped_children:
            branch_logic.setdefault(parent, {})[val] = skipped_children

    # NUMERIC detection
    if is_numeric_series(df[parent]):
        numeric_vals = pd.to_numeric(df[parent].replace('Unknown', np.nan), errors='coerce')
        for child in quiz_cols:
            if child == parent:
                continue
            dep_missing_mask = (df[child].astype(str).str.lower() == 'unknown') | (df[child].astype(str).str.lower() == 'skipped')
            if dep_missing_mask.sum() == 0:
                continue
            # look at numeric parent values where child is missing
            numeric_when_missing = numeric_vals[dep_missing_mask.values].dropna()
            if len(numeric_when_missing) < max(3, 0.01 * len(df)):
                continue
            threshold = numeric_when_missing.mean()
            # heuristics: consider '<threshold' if mean missing is lower than overall mean
            overall_mean = numeric_vals.mean()
            if np.isnan(overall_mean):
                continue
            if threshold < overall_mean:
                cond = f"<{threshold:.2f}"
            else:
                cond = f">{threshold:.2f}"
            branch_logic.setdefault(parent, {})[cond] = branch_logic.get(parent, {}).get(cond, []) + [child]

print(f"Auto-detected branching logic for {len(branch_logic)} parent questions.")

# ----------------------------
# Apply branching logic to mark Skipped questions
# (multi-layer: repeating once is usually enough; for nested layers we can iterate a few times)
# ----------------------------
def apply_branching(df, logic, iterations=2):
    df = df.copy()
    for _ in range(iterations):
        newly_skipped = 0
        for parent, conditions in logic.items():
            if parent not in df.columns:
                continue
            parent_s = df[parent].astype(str)
            for trigger, children in conditions.items():
                if isinstance(trigger, str) and (trigger.startswith('<') or trigger.startswith('>')):
                    # numeric threshold
                    try:
                        op = trigger[0]
                        val = float(trigger[1:])
                        mask = pd.to_numeric(parent_s.replace('Unknown', np.nan), errors='coerce')
                        if op == '<':
                            mask = mask < val
                        else:
                            mask = mask > val
                        mask = mask.fillna(False)
                    except Exception:
                        continue
                else:
                    mask = parent_s.str.lower() == str(trigger).lower()
                for child in children:
                    if child in df.columns:
                        # Only set to 'Skipped' if it's currently Unknown or blank or something not filled
                        before_mask_count = (df.loc[mask, child].astype(str).str.lower() == 'skipped').sum()
                        df.loc[mask & (df[child].astype(str).str.lower() != 'skipped'), child] = 'Skipped'
                        after_mask_count = (df.loc[mask, child].astype(str).str.lower() == 'skipped').sum()
                        newly_skipped += (after_mask_count - before_mask_count)
        if newly_skipped == 0:
            break
    return df

df = apply_branching(df, branch_logic, iterations=3)
print("Applied branching logic. Sample skipped columns (if any):", 
      [c for c in df.columns if (df[c].astype(str).str.lower() == 'skipped').any()][:10])

# ----------------------------
# Create smoothed probability columns (Laplace smoothing)
# Do this vectorized to avoid fragmenting the DataFrame.
# ----------------------------
smoothed_frames = []
for col in quiz_cols:
    # get value counts
    vc = df[col].astype(str).value_counts(dropna=False)
    categories = vc.index.astype(str).tolist()
    counts = vc.values.astype(float)
    K = len(categories)
    total = counts.sum()
    # Laplace: (count + 1) / (total + K)
    smoothed = (counts + 1.0) / (total + K)
    # map back to a Series aligned with df
    s = df[col].astype(str).map(dict(zip(categories, smoothed)))
    s.name = f"{col}_smoothed_prob"
    smoothed_frames.append(s)

if smoothed_frames:
    smoothed_df = pd.concat(smoothed_frames, axis=1)
    df = pd.concat([df, smoothed_df], axis=1)

# ----------------------------
# One-hot encode quiz response columns
# Exclude likely personal/product columns from encoding
encode_cols = [c for c in quiz_cols if not is_probable_personal_col(c) and not is_product_col(c)]
print(f"Encoding {len(encode_cols)} quiz columns into dummies.")
df_encoded = pd.get_dummies(df, columns=encode_cols, dummy_na=False)

# ----------------------------
# Product-category audit (attempt to locate product and category columns)
# ----------------------------
product_candidates = [c for c in df_encoded.columns if 'mxt_sku' in c.lower() or 'product' in c.lower() or 'shopify' in c.lower()]
category_candidates = [c for c in df_encoded.columns if 'category' in c.lower() or 'question_category' in c.lower()]

product_col = product_candidates[0] if product_candidates else None
category_col = category_candidates[0] if category_candidates else None

if product_col and category_col:
    # convert tuples/JSON strings back to readable strings for the audit
    audit_pairs = df_encoded[[product_col, category_col]].astype(str)
    duplicates = audit_pairs.duplicated(keep=False)
    if duplicates.any():
        print("Warning: duplicate product-category mappings found (sample):")
        print(audit_pairs.loc[duplicates].head(20))
    else:
        print("No duplicate product-category mappings found.")
else:
    print("Product or Category column not found for audit (skipping).")

# ----------------------------
# High-variance / low-sample detection (numeric only)
# ----------------------------
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    var_series = df_encoded[numeric_cols].var()
    uniq_series = df_encoded[numeric_cols].nunique()
    high_variance_cols = var_series[var_series > 0.25].index.tolist()
    low_sample_cols = uniq_series[uniq_series <= 1].index.tolist()
else:
    high_variance_cols = []
    low_sample_cols = []

print("High variance numeric columns:", high_variance_cols)
print("Low sample numeric columns:", low_sample_cols)

# ----------------------------
# Save processed CSV
# ----------------------------
df_encoded.to_csv(PROCESSED_CSV, index=False)
print("Saved processed CSV to:", PROCESSED_CSV)

# ----------------------------
## ----------------------------
# Data dictionary (safe for list-like or tuple-like values)
# ----------------------------
data_dict = []

for c in df_encoded.columns:
    col_series = df_encoded[c].dropna()

    # Get sample values safely (up to 5)
    try:
        unique_vals = pd.Series(col_series.unique()).astype(str)
        sample_vals = list(unique_vals[:5])
    except Exception:
        sample_vals = list(col_series.astype(str).head(5))

    # Safe num_unique: handle lists/tuples by converting to string
    try:
        num_unique = pd.Series(col_series.astype(str)).nunique()
    except Exception:
        # fallback: use set
        num_unique = len(set([str(v) for v in col_series]))

    data_dict.append({
        "column": c,
        "dtype": str(df_encoded[c].dtypes),
        "num_unique": num_unique,
        "sample_values": "; ".join(sample_vals)
    })

data_dict_df = pd.DataFrame(data_dict)
data_dict_df.to_csv(DATA_DICT_CSV, index=False)
print("Saved data dictionary to:", DATA_DICT_CSV)



# ----------------------------
# Dependency tree visualization
# ----------------------------
def plot_dependency_tree(branch_logic_map, output_png):
    G = nx.DiGraph()
    for parent, conds in branch_logic_map.items():
        for cond, children in conds.items():
            for child in children:
                # label the parent node with condition in edge
                G.add_edge(parent, child, label=str(cond))
    if G.number_of_nodes() == 0:
        print("No branching graph to plot.")
        return
    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(G, k=0.7, iterations=200, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightcyan", font_size=9, arrowsize=20)
    # draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Quiz Branching Dependency Tree")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print("Saved branching dependency tree to:", output_png)

plot_dependency_tree(branch_logic, BRANCH_TREE_PNG)

print("✅ dataprep_final completed successfully.")
