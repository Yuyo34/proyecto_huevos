import pandas as pd, numpy as np, re, os

KW_PRED = ["forecast","yhat","pred","prediction","y_pred","fcst","hat","value_pred"]
KW_DATE = ["date","ds","fecha"]
KW_ACT  = ["y","actual","target","obs","real","value_true","y_true","value","actuals"]

def pick_date(df):
    for c in df.columns:
        if any(k in c.lower() for k in KW_DATE):
            try:
                pd.to_datetime(df[c]); return c
            except Exception: pass
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except Exception: pass
    return df.columns[0]

def pick_pred(df):
    for c in df.columns:
        if any(k in c.lower() for k in KW_PRED): return c
    for c in df.columns:
        if re.search(r"(hat|pred|fcst)$", c.lower()): return c
    return None

def pick_actual(df, pred_col):
    meta = {"phase","split","is_test","is_train","is_valid","seasonal","trend","resid"}
    for c in df.columns:
        cl = c.lower()
        if c == pred_col or cl in meta: continue
        if any(k == cl for k in KW_ACT) or any(k in cl for k in KW_ACT): return c
    numc = [c for c in df.columns if c != pred_col and c.lower() not in meta]
    numc = [c for c in numc if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.7]
    if numc:
        vc = [(c, pd.to_numeric(df[c], errors="coerce").var()) for c in numc]
        vc.sort(key=lambda x: (np.nan_to_num(x[1], nan=-1.0)), reverse=True)
        return vc[0][0]
    return None

def metrics(path, m=12):
    df = pd.read_csv(path)
    dcol = pick_date(df)
    pcol = pick_pred(df)
    acol = pick_actual(df, pcol) if pcol else None
    if not pcol or not acol:
        raise SystemExit(f"No pude detectar columnas en {path}. date={dcol}, pred={pcol}, actual={acol}. Cols={list(df.columns)}")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.sort_values(dcol)
    y   = pd.to_numeric(df[acol], errors="coerce").to_numpy()
    yhat= pd.to_numeric(df[pcol], errors="coerce").to_numpy()
    mask= ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]
    if len(y) <= m:
        mase = np.nan
    else:
        den = np.mean(np.abs(y[m:] - y[:-m]))
        mase = float(np.mean(np.abs(y[m:] - yhat[m:]))/den) if den else np.nan
    mape = float(np.mean(np.abs((y - yhat)/y))) * 100
    smape= float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)))) * 100
    return dict(file=os.path.basename(path), date_col=dcol, actual_col=acol, pred_col=pcol,
                MASE=round(mase,3), MAPE=round(mape,1), sMAPE=round(smape,1),
                improv_pct=round((1 - mase)*100,1) if mase==mase else np.nan)

rows=[]
for p in ["out/fcst_base_h1.csv","out/fcst_base_h2.csv"]:
    rows.append(metrics(p))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("out/metrics_by_h.csv", index=False)
print("\nOK -> out/metrics_by_h.csv")
