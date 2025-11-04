# estimate_tau.py
# Run: python estimate_tau.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
import joblib
from preprocess import load_and_clean, feature_engineer

CSV = "cleaned_full.csv"   # change
TARGET="OVERALL LITERACY"
TREATMENT_COL="TOTAL_TEACHERS"  # continuous treatment

def prepare():
    df = load_and_clean(CSV)
    df = feature_engineer(df)
    # keep rows with non-null target and treatment
    df = df[pd.notna(df[TARGET]) & pd.notna(df[TREATMENT_COL])]
    # features X exclude target and treatment and id columns
    exclude = [TARGET, TREATMENT_COL, "STATE NAME", "DISTRICT NAME"]
    Xcols = [c for c in df.columns if c not in exclude]
    X = df[Xcols].select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    Y = pd.to_numeric(df[TARGET], errors="coerce").fillna(np.nan)
    T = pd.to_numeric(df[TREATMENT_COL], errors="coerce").fillna(np.nan)
    return df.reset_index(drop=True), X, T, Y, Xcols

def dml_residualize(X, Z, model=RandomForestRegressor(n_estimators=100, random_state=1)):
    # fit model Z ~ X and return residuals Z - E[Z|X]
    model.fit(X, Z)
    pred = model.predict(X)
    return Z - pred, model

def estimate_tau():
    df, X, T, Y, Xcols = prepare()
    # split for cross-fitting
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    tau_hat = np.zeros(len(df))
    # models used for nuisance
    model_t = RandomForestRegressor(n_estimators=200, random_state=1)
    model_y = RandomForestRegressor(n_estimators=200, random_state=2)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        T_tr, T_te = T.iloc[train_idx], T.iloc[test_idx]
        Y_tr, Y_te = Y.iloc[train_idx], Y.iloc[test_idx]
        # fit nuisances on train
        model_t.fit(X_tr, T_tr)
        model_y.fit(pd.concat([X_tr, T_tr.rename("T")], axis=1), Y_tr)
        # predict on test
        t_hat = model_t.predict(X_te)
        y_hat = model_y.predict(pd.concat([X_te, T_te.rename("T")], axis=1))
        # residuals
        r_t = T_te - t_hat
        r_y = Y_te - y_hat
        # orthogonalized simple estimator: regress r_y on r_t (no intercept) to get local slope per test sample
        # Use local linearization: slope = cov(r_y, r_t) / var(r_t) globally; here we compute per-fold slope and apply to test rows
        denom = np.sum(r_t ** 2)
        slope = np.sum(r_t * r_y) / (denom + 1e-8)
        # per-row tau approx = slope (constant for fold) but we can also multiply local scaling using variance of X
        tau_hat[test_idx] = slope
    df["tau_per_teacher"] = tau_hat
    # save
    df_out = df[["STATE NAME","DISTRICT NAME","tau_per_teacher"]].copy()
    df_out.to_csv("tau_estimates.csv", index=False)
    joblib.dump({"df":df, "Xcols":Xcols}, "dml_data.pkl")
    print("Saved tau_estimates.csv and dml_data.pkl")
    return df

if __name__=="__main__":
    estimate_tau()
