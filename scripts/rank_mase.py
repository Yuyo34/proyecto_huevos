import argparse, re, sys
import numpy as np, pandas as pd
from pathlib import Path

ACTUAL_CANDIDATES = ["actual","y","target","value_true","y_true","obs","real"]
PRED_CANDIDATES   = ["forecast","yhat","pred","prediction","value_pred","y_pred","fcst","hat"]
SPLIT_HINTS       = ["split","is_test","is_valid","phase"]  # si existe, ayuda a separar train/test

def mase(y, yhat, m=1):
    # Alinea por índice
    df = pd.concat({"y": y, "yhat": yhat}, axis=1).dropna()
    if len(df) <= m+1: 
        return np.nan
    denom = np.mean(np.abs(df["y"].values[m:] - df["y"].values[:-m]))
    if denom == 0:
        return np.nan
    num = np.mean(np.abs(df["y"].values[m:] - df["yhat"].values[m:]))
    return float(num/denom)

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in df.columns:
        if c.lower() in candidates:
            return c
    # tolerante a mayúsculas/espacios
    for want in candidates:
        for k,v in cols.items():
            if want == k.replace(" ",""):
                return v
    return None

def mase_from_csv(p, m):
    try:
        df = pd.read_csv(p, parse_dates=True)
    except Exception:
        return None
    # busca par actual/pred
    a = find_col(df, ACTUAL_CANDIDATES)
    f = find_col(df, PRED_CANDIDATES)
    if not a or not f:
        return None
    # intenta usar índice temporal si existe col "date"
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        except Exception:
            pass

    # Si hay columna de split, usa sólo filas marcadas como test/valid
    split_col = None
    for c in SPLIT_HINTS:
        if c in (x.lower() for x in df.columns):
            # encuentra el nombre real respetando mayúsculas
            split_col = [x for x in df.columns if x.lower()==c][0]
            break

    use = df.copy()
    if split_col is not None:
        mask = use[split_col].astype(str).str.lower().isin(["test","valid","validation","1","true","eval"])
        if mask.any():
            use = use[mask]

    y    = pd.to_numeric(use[a], errors="coerce")
    yhat = pd.to_numeric(use[f], errors="coerce")
    return mase(y, yhat, m=m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs", help="Carpeta con logs")
    ap.add_argument("--outdir", default="out", help="Carpeta con CSVs")
    ap.add_argument("-m","--seasonality", type=int, default=12, help="Estacionalidad para MASE (mensual=12)")
    args = ap.parse_args()

    rows = []

    # 1) Logs: buscar MASE=...
    logdir = Path(args.logdir)
    if logdir.exists():
        for p in list(logdir.glob("*.log")) + list(logdir.glob("*.txt")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"MASE\s*=\s*([0-9.]+)", txt, flags=re.IGNORECASE)
            if m:
                try:
                    rows.append({"source": f"log:{p.name}", "mase": float(m.group(1))})
                except Exception:
                    pass

    # 2) CSVs: intentar calcular MASE si hay columnas reconocibles
    outdir = Path(args.outdir)
    if outdir.exists():
        for p in outdir.glob("*.csv"):
            val = mase_from_csv(p, m=args.seasonality)
            if val is not None and np.isfinite(val):
                rows.append({"source": f"csv:{p.name}", "mase": float(val)})

    if not rows:
        print("No se detectaron MASE en logs ni se pudo calcular desde CSVs.")
        print("Sugerencias:")
        print("  a) Asegúrate de guardar logs con 'MASE=...'. Ejemplo PowerShell:")
        print("     $ts = Get-Date -Format \"yyyyMMdd_HHmmss\"")
        print("     python -m upgrade_precision.pipeline.pipeline_monthly_exog ... 2>&1 | Tee-Object -FilePath (\"logs\\run_$ts.log\")")
        print("  b) O incluye columnas (actual, forecast) o (y, yhat) en los CSV de out\\ para que el script calcule MASE.")
        sys.exit(0)

    rank = pd.DataFrame(rows).sort_values("mase").reset_index(drop=True)
    print(rank.to_string(index=False))
    outdir.mkdir(parents=True, exist_ok=True)
    rank.to_csv(outdir / "mase_rankings.csv", index=False, encoding="utf-8")
    print("\nOK -> " + str(outdir / "mase_rankings.csv"))

if __name__ == "__main__":
    main()
