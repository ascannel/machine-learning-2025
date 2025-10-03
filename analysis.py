import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

ID_NAME_HINTS = re.compile(
    r'(?:^id$|_id$|^hash$|^sha1$|^sha256$|^guid$|^file(?:name)?$|^Unnamed:\s*0$)',
    re.I
)

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.txt'):
        return pd.read_csv(path, low_memory=False)
    if ext in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    if ext in ('.json', '.jsonl'):
        return pd.read_json(path, lines=ext == '.jsonl')
    # Last resort: try pandas magic
    return pd.read_csv(path, low_memory=False)

def coerce_numeric_like(s: pd.Series) -> pd.Series:
    """Попытаться превратить строковые значения с валютой/разделителями в числа."""
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        cleaned = (
            s.astype(str)
             .str.replace(r'[^0-9eE+\-\.]', '', regex=True)
             .replace('', pd.NA)
        )
        return pd.to_numeric(cleaned, errors='coerce')
    return pd.to_numeric(s, errors='coerce')

def suggest_target_column(df: pd.DataFrame):
    # try common names first
    common = ['target', 'label', 'class', 'y', 'Category', 'category', 'Class']
    for c in common:
        if c in df.columns:
            return c
    # fallback: pick object/category with small number of unique values
    candidates = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nunq = s.nunique(dropna=True)
        if pd.api.types.is_categorical_dtype(s) or s.dtype == 'object':
            if 1 < nunq <= min(100, max(2, int(0.2 * n))):
                candidates.append((col, nunq))
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0] if candidates else None

def split_features_target(df: pd.DataFrame, target_col: str | None):
    df = df.copy()

    id_cols = [c for c in df.columns if ID_NAME_HINTS.match(str(c))]
    n = len(df)
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        nunq = s.nunique(dropna=True)
        if n > 0 and nunq / n >= 0.95:
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s):
                if col not in id_cols:
                    id_cols.append(col)

    if target_col and target_col in id_cols:
        id_cols.remove(target_col)

    X = df.drop(columns=id_cols, errors='ignore')
    if target_col and target_col in X.columns:
        y = X.pop(target_col)
    else:
        y = None

    return X, y, id_cols

def classify_object_series(s: pd.Series, n_rows: int):
    """Return 'text' or 'categorical' for object dtype."""
    non_na = s.dropna().astype(str)
    if non_na.empty:
        return 'categorical'  # degenerate
    avg_len = non_na.str.len().mean()
    avg_tokens = non_na.str.split().map(len).mean()
    uniq_ratio = non_na.nunique() / max(1, len(non_na))
    if (avg_len >= 30 or avg_tokens >= 5) and uniq_ratio > 0.2:
        return 'text'
    nunq = non_na.nunique()
    if nunq <= min(50, int(0.2 * n_rows)):
        return 'categorical'

    return 'other'

def detect_feature_types(X: pd.DataFrame):
    n = len(X)
    types = {
        'numeric': [],
        'categorical': [],
        'text': [],
        'datetime': [],
        'boolean': [],
        'other': [],
    }

    for col in X.columns:
        s = X[col]

        if pd.api.types.is_bool_dtype(s):
            types['boolean'].append(col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            types['numeric'].append(col)
            continue

        if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
            num_try = coerce_numeric_like(s)
            if num_try.notna().mean() > 0.9:
                types['numeric'].append(col)
                continue

            sample = pd.to_datetime(
                s.sample(min(500, len(s))), errors='coerce', utc=True
            )
            if sample.notna().mean() > 0.9:
                types['datetime'].append(col)
                continue

            cls = classify_object_series(s, n_rows=n)
            (types[cls] if cls in types else types['other']).append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            types['datetime'].append(col)
            continue

        types['other'].append(col)

    return types

def overall_missing_pct(df: pd.DataFrame) -> float:
    total = df.size
    if total == 0:
        return 0.0
    return float(df.isna().sum().sum()) / total * 100.0

def outlier_row_pct_numeric(X_num: pd.DataFrame, iqr_k=1.5) -> float:
    if X_num.empty:
        return 0.0
    outlier_any = pd.Series(False, index=X_num.index)
    for col in X_num.columns:
        x = pd.to_numeric(X_num[col], errors='coerce')
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        mask = (x < (q1 - iqr_k * iqr)) | (x > (q3 + iqr_k * iqr))
        outlier_any = outlier_any | mask.fillna(False)
    return outlier_any.mean() * 100.0

def class_distribution(y: pd.Series):
    cnt = y.value_counts(dropna=False)
    pct = (cnt / cnt.sum() * 100).round(2)
    # compressed min/max view like "5/95"
    if len(cnt) >= 2:
        mins = pct.min()
        maxs = pct.max()
        ratio = f"{mins:.0f}/{maxs:.0f}"
    else:
        ratio = "—"
    return cnt.to_dict(), pct.to_dict(), ratio

def main():
    ap = argparse.ArgumentParser(description="Dataset profiler")
    ap.add_argument("--file", required=True, help="Path to main CSV/Parquet")
    ap.add_argument("--target", default=None, help="Target/label column (optional)")
    ap.add_argument("--imbalance-threshold", type=float, default=0.80,
                    help="Max class share to flag imbalance (default 0.80 = 80%)")
    ap.add_argument("--iqr-k", type=float, default=1.5,
                    help="IQR coefficient for outliers (default 1.5)")
    ap.add_argument("--save", default="dataset_profile.json", help="Where to save JSON report")
    args = ap.parse_args()

    df = load_table(args.file)

    # Guess target if not provided
    target = args.target or suggest_target_column(df)

    X, y, dropped_ids = split_features_target(df, target)
    types = detect_feature_types(X)

    # Counts
    N = len(df)
    d = X.shape[1]

    # Missingness
    missing_pct_all = overall_missing_pct(df)
    missing_pct_features = overall_missing_pct(X)

    # Outliers
    X_num = X[types['numeric']]
    outlier_pct_rows = outlier_row_pct_numeric(X_num, iqr_k=args.iqr_k)

    # Class stats (for classification only)
    K = None
    class_cnt = class_pct = class_ratio = None
    imbalanced_flag = None
    if y is not None:
        # Heuristic: if numeric with many unique -> treat as regression
        if (pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(20, int(0.1 * len(y)))):
            K = "— (regression?)"
            imbalanced_flag = "—"
        else:
            K = int(y.nunique(dropna=True))
            cnt, pct, ratio = class_distribution(y)
            class_cnt, class_pct, class_ratio = cnt, pct, ratio
            max_share = max(pct.values()) if len(pct) else 0
            imbalanced_flag = "да" if (max_share / 100.0) >= args.imbalance_threshold else "нет"
    else:
        K = "— (target not set)"
        imbalanced_flag = "—"

    # Feature type counts
    type_counts = {k: len(v) for k, v in types.items()}
    other_types_present = {k: v for k, v in type_counts.items() if k not in ('numeric', 'categorical', 'text', 'datetime', 'boolean') and v > 0}

    # Binary yes/no flags
    flags = {
        "Наличие пропущенных значений (да/нет)": "да" if missing_pct_all > 0 else "нет",
        "Разнородные признаки (да/нет)": "да" if sum(1 for k in ['numeric','categorical','text','datetime','boolean'] if type_counts[k] > 0) >= 2 else "нет",
        "Несбалансированные классы (да/нет)": imbalanced_flag,
        "Большое количество выбросов (да/нет)": "да" if outlier_pct_rows >= 5.0 else "нет",
        "Наличие текстовых признаков (да/нет)": "да" if type_counts['text'] > 0 else "нет",
    }

    report = {
        "dataset_path": os.path.abspath(args.file),
        "target_column": target,
        "dropped_id_like_columns": dropped_ids,
        "N (rows)": N,
        "d (features)": d,
        "K (classes)": K,
        "missing_pct_overall": round(missing_pct_all, 3),
        "missing_pct_features_only": round(missing_pct_features, 3),
        "class_counts": class_cnt,
        "class_percents": class_pct,
        "class_ratio_min_max": class_ratio,
        "outlier_rows_pct_IQR": round(outlier_pct_rows, 3),
        "feature_type_counts": type_counts,
        "other_types": other_types_present,
        "flags": flags,
        "params": {
            "imbalance_threshold": args.imbalance_threshold,
            "iqr_k": args.iqr_k
        },
        "columns_by_type": types,
        "other_columns": types.get('other', [])
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved report → {args.save}")

if __name__ == "__main__":
    main()