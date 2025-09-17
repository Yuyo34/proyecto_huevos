import pandas as pd, numpy as np, os
from statsmodels.tsa.statespace.sarimax import SARIMAX

M = 12
BT_INIT = 12
ORDER = (1,0,0)
SORDER= (0,1,0,M)

TARGET = "data/precio_huevo_mensual_real_FOR_ROCV.csv"
EXOGS  = [
    ("data/usdclp_dlog.csv", "usd"),
    ("data/imacec_yoy.csv",  "ipc"),
    ("data/pct_imp_yoy_FOR_TARGET.csv", "pctimp"),
]

def read_series(path, name):
    df = pd.read_csv(path)
    dcol = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    vcol = next((c for c in df.columns if c != dcol), None)
    if vcol is None: raise SystemExit(f"No encuentro columna de valor en {path}. Cols={list(df.columns)}")
    dt = pd.to_datetime(df[dcol], errors="coerce")
    val= pd.to_numeric(df[vcol], errors="coerce")
    out= pd.DataFrame({"date": dt, name: val}).dropna()
    out["month"] = out["date"].dt.to_period("M")
    out = out.drop(columns=["date"]).drop_duplicates("month").set_index("month")
    return out

def build_panel():
    y = read_series(TARGET, "y")
    Xs = [read_series(p, nm) for p,nm in EXOGS if os.path.exists(p)]
    idx_full = pd.period_range(y.index.min(), y.index.max(), freq="MS")
    df = pd.DataFrame(index=idx_full).join(y, how="left")
    for x in Xs: df = df.join(x, how="left")
    for c in df.columns:
        if c != "y": df[c] = df[c].ffill().bfill()
    # recorte a bloque contiguo
    y_notna = df["y"].notna().astype(int)
    start = int(np.argmax(y_notna.values)) if y_notna.any() else 0
    end   = len(y_notna) - int(np.argmax(y_notna.values[::-1]))
    df = df.iloc[start:end].dropna(subset=["y"])
    df.index = df.index.to_timestamp("M")
    return df

def seasonal_denom_calendar(y_series, m=M):
    """Denominador MASE: promedio |y_t - y_{t-12}| usando calendario mensual con reindex completo."""
    s = pd.Series(y_series, index=y_series.index)
    # full monthly index
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s_full = s.reindex(full_idx)
    diffs = (s_full - s_full.shift(m)).abs().dropna()
    return float(diffs.mean()), int(diffs.notna().sum())

def rolling_origin(df, h):
    y = df["y"]
    X = df.drop(columns=["y"])
    preds = []
    for t in range(BT_INIT, len(df)-h+1):
        y_tr, X_tr = y.iloc[:t], X.iloc[:t]
        X_fc = X.iloc[t:t+h]
        try:
            mod = SARIMAX(y_tr, exog=X_tr, order=ORDER, seasonal_order=SORDER,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            y_hat_h = float(res.get_forecast(steps=h, exog=X_fc).predicted_mean.iloc[-1])
        except Exception:
            y_hat_h = float(y_tr.iloc[-M] if len(y_tr) > M else y_tr.iloc[-1])
        target_date = y.index[t+h-1]
        preds.append((target_date, float(y.loc[target_date]), y_hat_h))
    pred_df = pd.DataFrame(preds, columns=["date","y_true","y_hat"]).set_index("date")
    return pred_df.reset_index()

def main():
    df = build_panel()
    print(f"Len(df)={len(df)}; rango= {df.index.min().date() if len(df) else None} -> {df.index.max().date() if len(df) else None}")

    rows = []
    for h in (1,2):
        pred = rolling_origin(df, h)
        pred.to_csv(f"out/rocv_h{h}_pred.csv", index=False)
        mae_h = float(np.mean(np.abs(pred["y_true"] - pred["y_hat"]))) if len(pred) else np.nan
        den, den_n = seasonal_denom_calendar(df["y"])
        mase = (mae_h/den) if (den and not np.isnan(den)) else np.nan
        rows.append({"h":h, "rows":len(pred), "MAE_h": round(mae_h,3) if mae_h==mae_h else None,
                     "den": round(den,3) if den==den else None, "den_pairs": den_n,
                     "MASE": round(mase,3) if mase==mase else None})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\nGuardado:\n - out/rocv_h1_pred.csv\n - out/rocv_h2_pred.csv")

if __name__ == "__main__":
    main()

