#!/usr/bin/env python3
# enhanced_preprocess.py
# Usage: python enhanced_preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.cwd()
IN_CSV = ROOT / "elementary_2015_16.csv"
OUT_CLEAN = ROOT / "cleaned_full.csv"
OUT_STRICT = ROOT / "cleaned_strict.csv"

# Configurable parameters
TARGET = "OVERALL LITERACY"
STUDENTS = "PRESTD"                # change if your student column has different name
TEACHER_CANDIDATES = [
    "PMTCH", "PFTCH", "TOT_TCH", "TEACHERS_MALE", "TEACHERS_FEMALE",
    "TOTAL_TEACHERS", "TOT_TCHS"
]
MIN_STUDENTS = 5                   # drop rows with fewer than this many students
SPARSE_COL_THRESHOLD = 0.80        # drop columns with >80% missing
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

def load_and_normalize(path):
    df = pd.read_csv(path, dtype=str)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # drop completely empty columns
    df = df.loc[:, df.notna().any(axis=0)]
    return df

def coerce_numbers(df):
    for c in df.columns:
        # remove commas, trim spaces then coerce
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip().str.replace(",", "").replace("", np.nan)
            # if many rows look numeric convert column
            numeric_frac = s.dropna().head(200).apply(lambda x: bool(pd.to_numeric(x, errors="coerce"))) .mean() if len(s.dropna())>0 else 0
            # simpler heuristic: try convert and check number of non-nulls
            conv = pd.to_numeric(s, errors="coerce")
            if conv.notna().sum() > 0:
                df[c] = conv
    return df

def create_total_teachers(df):
    # if TOTAL_TEACHERS already present, keep it
    if "TOTAL_TEACHERS" in df.columns:
        df["TOTAL_TEACHERS"] = pd.to_numeric(df["TOTAL_TEACHERS"], errors="coerce")
        return df
    # find candidate teacher columns in the dataset
    present = [c for c in TEACHER_CANDIDATES if c in df.columns]
    if present:
        df["TOTAL_TEACHERS"] = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        return df
    # If no teacher columns, try to infer from school + staff columns (common patterns)
    possible = [c for c in df.columns if any(k in c.upper() for k in ["TCH","TEACH","STAFF","TOT"])]
    if possible:
        # best-effort: sum numeric-looking possible columns (this is a fallback)
        nums = []
        for c in possible:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                nums.append(c)
        if nums:
            df["TOTAL_TEACHERS"] = df[nums].fillna(0).sum(axis=1)
            return df
    # if nothing found, create TOTAL_TEACHERS as NaN and caller will handle
    df["TOTAL_TEACHERS"] = np.nan
    return df

def feature_engineer(df):
    # students numeric
    if STUDENTS in df.columns:
        df[STUDENTS] = pd.to_numeric(df[STUDENTS], errors="coerce")
    # students_per_teacher
    df["students_per_teacher"] = df.apply(
        lambda r: r[STUDENTS] / r["TOTAL_TEACHERS"]
        if pd.notna(r.get(STUDENTS)) and pd.notna(r.get("TOTAL_TEACHERS")) and r["TOTAL_TEACHERS"]>0
        else np.nan,
        axis=1
    )
    return df

def drop_sparse_and_bad_rows(df):
    # drop sparse columns
    miss_frac = df.isna().mean()
    drop_cols = miss_frac[miss_frac > SPARSE_COL_THRESHOLD].index.tolist()
    if drop_cols:
        print(f"Dropping {len(drop_cols)} sparse columns (>{SPARSE_COL_THRESHOLD*100:.0f}% missing).")
        df = df.drop(columns=drop_cols)
    # drop rows missing essential fields
    essential = []
    if TARGET in df.columns:
        essential.append(TARGET)
    essential.append("TOTAL_TEACHERS")
    df = df.dropna(subset=essential, how="any")
    # remove rows with implausible students
    if STUDENTS in df.columns:
        df = df[df[STUDENTS].fillna(0) >= MIN_STUDENTS]
    return df

def impute_and_winsorize(df):
    # choose numeric columns for imputation (exclude TARGET and treatment)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [TARGET, "TOTAL_TEACHERS"]]
    # median impute
    med = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(med)
    # for likely binary indicators (values 0/1) fillna 0
    for c in num_cols:
        vals = pd.Index(df[c].dropna().unique())
        if vals.isin([0,1]).all():
            df[c] = df[c].fillna(0)
    # winsorize selected numeric columns
    if "students_per_teacher" in df.columns:
        low = df["students_per_teacher"].quantile(WINSOR_LOWER)
        high = df["students_per_teacher"].quantile(WINSOR_UPPER)
        df["students_per_teacher"] = df["students_per_teacher"].clip(low, high)
    return df

def save_outputs(df):
    df.to_csv(OUT_CLEAN, index=False)
    modelling_cols = [TARGET, "TOTAL_TEACHERS", "students_per_teacher"]
    modelling_cols = [c for c in modelling_cols if c in df.columns]
    strict = df.dropna(subset=modelling_cols)
    strict.to_csv(OUT_STRICT, index=False)
    print(f"Wrote {OUT_CLEAN.name} ({len(df)} rows) and {OUT_STRICT.name} ({len(strict)} rows)")

def main():
    print("Loading:", IN_CSV)
    df = load_and_normalize(IN_CSV)
    df = coerce_numbers(df)
    df = create_total_teachers(df)
    df = feature_engineer(df)
    df = drop_sparse_and_bad_rows(df)
    df = impute_and_winsorize(df)
    save_outputs(df)

def load_and_clean(path):
    df = load_and_normalize(path)
    df = coerce_numbers(df)
    df = create_total_teachers(df)
    df = feature_engineer(df)
    df = drop_sparse_and_bad_rows(df)
    df = impute_and_winsorize(df)
    return df


if __name__ == "__main__":
    main()
