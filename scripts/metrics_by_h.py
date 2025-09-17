import pandas as pd, numpy as np, os

def load_cols(df):
    t = next(c for c in df.columns if c.lower() in ["date","ds"])
    y = next(c for c in df.columns if c.lower() in ["y","actual","target","obs","real","value_true"])
    ph= next(c for c in df.columns if c.lower() in ["forecast","yhat","pred","prediction","y_pred","fcst","hat","value_pred"])
    return t,y,ph

def metrics(path, m=12):
    df = pd.read_csv(path)
    t,yc,pc = load_cols(df)
    df[t] = pd.to_datetime(df[t]); df = df.sort_values(t)
    y, yhat = df[yc].to_numpy(), df[pc].to_numpy()
    # filtra pares válidos
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]
    den = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
    mase = float(np.mean(np.abs(y[m:] - yhat[m:]))/den) if den and len(y)>m else np.nan
    mape = float(np.mean(np.abs((y - yhat)/y))) * 100
    smape= float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)))) * 100
    return dict(file=os.path.basename(path), MASE=round(mase,3), MAPE=round(mape,1), sMAPE=round(smape,1),
                improv_pct=round((1 - mase)*100,1))

rows = []
for p in ["out/fcst_base_h1.csv","out/fcst_base_h2.csv"]:
    rows.append(metrics(p))

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("out/metrics_by_h.csv", index=False)
print("\nOK -> out/metrics_by_h.csv")
