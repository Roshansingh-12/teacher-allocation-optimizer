# data_inspect.py
# Run: python data_inspect.py
import pandas as pd
import json
CSV="elementary_2015_16.csv"  # change to elementary_2015_16.csv if needed
def summarize(path, nrows=500):
    df = pd.read_csv(path, nrows=nrows)
    out=[]
    for c in df.columns:
        col=df[c]
        sample=list(col.dropna().unique()[:6])
        out.append({
            "col": c,
            "dtype": str(col.dtype),
            "non_null": int(col.count()),
            "unique_sample": sample,
            "unique_count_est": int(col.nunique(dropna=True))
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__=="__main__":
    summarize(CSV)
