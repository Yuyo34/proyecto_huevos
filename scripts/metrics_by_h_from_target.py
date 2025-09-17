import pandas as pd, numpy as np, os

TARGET_PATH = "data/precio_huevo_mensual_real.csv"  # ajusta si tu target se llama distinto

def read_target(path):
    df = pd.read_csv(path)
    # detectar columnas de fecha y valor
    dcol = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    vcol_opts = ["y","value","actual","target","obs","real","y_true","actuals"]
    vcol = next((c for c in df.columns if c.lower() in vcol_opts), None)
    if vcol is None:
        # primera numérica que no sea date
        num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.7]
        if not num: raise SystemExit(f"No pude detectar columna de valor en {path}. Cols={list(df.columns)}")
        vcol = num[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df[[dcol, vcol]].rename(columns={dcol:"date", vcol:"y"}).dropna()

def read_fcst(path):
    df = pd.read_csv(path)
    dcol = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    pcol_opts = ["forecast","yhat","pred","prediction","y_pred","fcst","hat","value_pred"]
    pcol = next((c for c in df.columns if c.lower() in pcol_opts), None)
    if pcol is None:
        # fallback: última numérica que no sea fecha
        num = [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.7]
        if not num: raise SystemExit(f"No pude detectar columna de forecast en {path}. Cols={list(df.columns)}")
        pcol = num[-1]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df[[dcol, pcol]].rename(columns={dcol:"date", pcol:"forecast"}).dropna()

def compute_metrics(y, yhat, m=12):
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat = y[mask], yhat[mask]
    # MASE
    if len(y) > m:
        den = np.mean(np.abs(y[m:] - y[:-m]))
        mase = float(np.mean(np.abs(y[m:] - yhat[m:]))/den) if den else np.nan
    else:
        mase = np.nan
    # MAPE (evitando dividir por 0)
    nz = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz]))) * 100 if nz.any() else np.nan
    # sMAPE
    smape = float(np.mean(2*np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)))) * 100
    return mase, mape, smape

def report(fc_path, target):
    fc = read_fcst(fc_path)
    merged = pd.merge(fc, target, on="date", how="inner").dropna()
    y, yhat = merged["y"].to_numpy(), merged["forecast"].to_numpy()
    mase, mape, smape = compute_metrics(y, yhat, m=12)
    return {
        "file": os.path.basename(fc_path),
        "rows": int(len(merged)),
        "date_min": merged["date"].min().date().isoformat() if len(merged) else None,
        "date_max": merged["date"].max().date().isoformat() if len(merged) else None,
        "MASE": round(mase, 3) if mase==mase else None,
        "MAPE": round(mape, 1) if mape==mape else None,
        "sMAPE": round(smape, 1) if smape==smape else None,
        "improv_pct": round((1 - mase)*100, 1) if mase==mase else None
    }

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
