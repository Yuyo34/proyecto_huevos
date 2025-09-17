import pandas as pd, numpy as np, os

def pick(df, kind):
    kw = dict(
        date=["date","ds","fecha"],
        pred=["forecast","yhat","pred","prediction","y_pred","fcst","hat","value_pred"],
        actual=["y","actual","target","obs","real","y_true","value","actuals","value_true"]
    )[kind]
    for c in df.columns:
        if c.lower() in kw: return c
    return None

def metrics(bt_path, m=12):
    df = pd.read_csv(bt_path)
    d = pick(df,"date") or df.columns[0]
    p = pick(df,"pred")
    a = pick(df,"actual")
    if not (p and a):
        raise SystemExit(f"No encuentro columnas en {bt_path}. Cols={list(df.columns)}")
    df[d] = pd.to_datetime(df[d], errors="coerce")
    df = df.sort_values(d)
    y   = pd.to_numeric(df[a], errors="coerce").to_numpy()
    yhat= pd.to_numeric(df[p], errors="coerce").to_numpy()
    mask= ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]
    den = np.mean(np.abs(y[m:] - y[:-m])) if len(y)>m else np.nan
    mase = float(np.mean(np.abs(y[m:] - yhat[m:]))/den) if den and len(y)>m else np.nan
    nz = np.abs(y)>1e-12
    mape = float(np.mean(np.abs((y[nz]-yhat[nz])/y[nz])))*100 if nz.any() else np.nan
    smape= float(np.mean(2*np.abs(y-yhat)/(np.abs(y)+np.abs(yhat))))*100
    return dict(file=os.path.basename(bt_path), rows=len(y),
                MASE=round(mase,3) if mase==mase else None,
                MAPE=round(mape,1) if mape==mape else None,
                sMAPE=round(smape,1) if smape==smape else None,
                improv_pct=round((1-mase)*100,1) if mase==mase else None)

rows=[]
for p in ["out/bt_base_h1.csv","out/bt_base_h2.csv"]:
    if os.path.exists(p):
        rows.append(metrics(p))
    else:
        rows.append(dict(file=os.path.basename(p), rows=0, MASE=None, MAPE=None, sMAPE=None, improv_pct=None))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("out/metrics_by_h.csv", index=False)
print("\nOK -> out/metrics_by_h.csv")
