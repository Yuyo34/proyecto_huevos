import pandas as pd, numpy as np, os

TARGET_PATH = "data/precio_huevo_mensual_real.csv"  # usa este target

def read_target(path):
    df = pd.read_csv(path)
    dcol = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    vcol = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["y","value","actual","target","obs","real","y_true","actuals"]:
            vcol = c; break
    if vcol is None:
        num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean()>0.7]
        if not num: raise SystemExit(f"No pude detectar columna de valor en {path}. Cols={list(df.columns)}")
        vcol = num[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df["month"] = df[dcol].dt.to_period("M")
    return df[["month", vcol]].rename(columns={vcol:"y"}).dropna()

def read_fcst(path):
    df = pd.read_csv(path)
    dcol = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    pcol = None
    for c in df.columns:
        if c.lower() in ["forecast","yhat","pred","prediction","y_pred","fcst","hat","value_pred"]:
            pcol = c; break
    if pcol is None:
        num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean()>0.7]
        if not num: raise SystemExit(f"No pude detectar columna de forecast en {path}. Cols={list(df.columns)}")
        pcol = num[-1]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df["month"] = df[dcol].dt.to_period("M")
    return df[["month", pcol]].rename(columns={pcol:"forecast"}).dropna()

def compute_metrics(y, yhat, m=12):
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]
    if len(y) > m:
        den = np.mean(np.abs(y[m:] - y[:-m]))
        mase = float(np.mean(np.abs(y[m:] - yhat[m:]))/den) if den else np.nan
    else:
        mase = np.nan
    nz = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz]))) * 100 if nz.any() else np.nan
    smape = float(np.mean(2*np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)))) * 100
    return mase, mape, smape

def report(fc_path, target):
    fc = read_fcst(fc_path)
    merged = pd.merge(fc, target, on="month", how="inner").dropna()
    out = {
        "file": os.path.basename(fc_path),
        "rows": int(len(merged)),
        "month_min": str(merged["month"].min()) if len(merged) else None,
        "month_max": str(merged["month"].max()) if len(merged) else None,
        "MASE": None, "MAPE": None, "sMAPE": None, "improv_pct": None
    }
    if len(merged):
        y, yhat = merged["y"].to_numpy(), merged["forecast"].to_numpy()
        mase, mape, smape = compute_metrics(y, yhat, m=12)
        out.update({
            "MASE": round(mase,3) if mase==mase else None,
            "MAPE": round(mape,1) if mape==mape else None,
            "sMAPE": round(smape,1) if smape==smape else None,
            "improv_pct": round((1 - mase)*100,1) if mase==mase else None
        })
    else:
        # Debug: guarda meses de cada lado
        fc.assign(side="forecast").to_csv("out/debug_months_fc_"+os.path.basename(fc_path)+".csv", index=False)
        target.assign(side="target").to_csv("out/debug_months_target.csv", index=False)
    return out

def main():
    target = read_target(TARGET_PATH)
    rows = []
    for p in ["out/fcst_base_h1.csv", "out/fcst_base_h2.csv"]:
        rows.append(report(p, target))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("out/metrics_by_h.csv", index=False)
    print("\nOK -> out/metrics_by_h.csv")

if __name__ == "__main__":
    main()
