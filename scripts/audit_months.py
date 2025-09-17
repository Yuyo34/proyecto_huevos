import pandas as pd, os

def monthly_gaps(path):
    df = pd.read_csv(path)
    dcol = next(c for c in df.columns if c.lower() in ["date","ds","fecha"])
    idx  = pd.to_datetime(df[dcol], errors="coerce").dt.to_period("M").dt.to_timestamp()
    full = pd.date_range(idx.min(), idx.max(), freq="MS")
    missing = full.difference(idx)
    return {
        "file": path, "range": f"{idx.min():%Y-%m} -> {idx.max():%Y-%m}",
        "rows": len(idx), "should_be": len(full), "missing": len(missing),
        "some_missing": [d.strftime("%Y-%m") for d in list(missing)[:12]]
    }

cands = [
    r"data\precio_huevo_mensual_real.csv",
    r"out\fcst_base_h1.csv", r"out\fcst_base_h2.csv",
    r"data\ipc_index.csv", r"data\ipc.csv", r"data\IPC.csv", r"data\ipc_level.csv"
]
for p in cands:
    if os.path.exists(p):
        print(monthly_gaps(p))
