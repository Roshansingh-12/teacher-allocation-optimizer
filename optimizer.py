# optimizer.py
# two modes:
#  - greedy (linear approx): sort by tau_per_teacher and allocate until budget used
#  - knapsack (diminishing returns): build discrete marginal gains per extra teacher and solve ILP with pulp
import pandas as pd
import numpy as np
import joblib

def load_tau(path="tau_estimates.csv"):
    return pd.read_csv(path)

def greedy_alloc(tau_df, budget, cap_per_district=None):
    df = tau_df.copy()
    df = df.sort_values("tau_per_teacher", ascending=False).reset_index(drop=True)
    alloc = np.zeros(len(df), dtype=int)
    remaining = int(budget)
    caps = [np.inf]*len(df) if cap_per_district is None else [cap_per_district.get(r["DISTRICT NAME"], cap_per_district.get(r["STATE NAME"], np.inf)) for _,r in df.iterrows()]
    for i, r in df.iterrows():
        if remaining<=0:
            break
        cap = int(caps[i]) if not np.isinf(caps[i]) else remaining
        take = min(cap, remaining)
        alloc[i]=take
        remaining -= take
    df["allocated_teachers"] = alloc
    df["predicted_gain_pct"] = df["allocated_teachers"] * df["tau_per_teacher"]
    return df

def knapsack_alloc(tau_df, budget, max_per=10):
    # compute diminishing returns per district by assuming simple concave function:
    # marginal_k = tau_i * (1 - decay*(k-1))
    # or use marginal = tau_i * exp(-beta * (k-1))
    df = tau_df.copy()
    items=[]
    for idx, r in df.iterrows():
        tau = float(r["tau_per_teacher"])
        # choose beta relative to tau: small decay for larger districts
        beta = 0.15
        for k in range(1, max_per+1):
            marginal = tau * np.exp(-beta*(k-1))
            items.append({"district_idx": idx, "district": r["DISTRICT NAME"], "value": marginal, "cost": 1, "k": k})
    # greedy selection on item value per cost (since cost=1, just sort by value)
    items_sorted = sorted(items, key=lambda x: -x["value"])
    alloc = {}
    remaining = int(budget)
    for it in items_sorted:
        if remaining<=0: break
        # accept item only if we haven't already assigned k-1 for that district
        curr = alloc.get(it["district_idx"], 0)
        if curr + 1 == it["k"]:
            alloc[it["district_idx"]] = curr + 1
            remaining -= 1
    # build result df
    alloc_list = [alloc.get(i,0) for i in range(len(df))]
    df["allocated_teachers"] = alloc_list
    df["predicted_gain_pct"] = df["allocated_teachers"] * df["tau_per_teacher"]  # approximate
    return df

if __name__=="__main__":
    tau_df = load_tau()
    print("Top districts by tau:")
    print(tau_df.sort_values("tau_per_teacher", ascending=False).head(10))
    # example usage
    out = greedy_alloc(tau_df, budget=50, cap_per_district=None)
    out.to_csv("allocation_greedy.csv", index=False)
    print("Wrote allocation_greedy.csv")
    out2 = knapsack_alloc(tau_df, budget=50, max_per=8)
    out2.to_csv("allocation_knapsack.csv", index=False)
    print("Wrote allocation_knapsack.csv")
