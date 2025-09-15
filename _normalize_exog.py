import sys, csv, pandas as pd, numpy as np
from pathlib import Path

"""
Uso:
  py _normalize_exog.py INPUT.csv OUTPUT.csv [--log]

- Autodetecta separador (coma/; | tab).
- Detecta columnas de fecha y valor (date/fecha/mes/month/period | value/precio/price/valor/index/indice/tipo_cambio/ipc/diesel/usd/corn/soy).
- Convierte a mensual (MS) promediando si la fuente es diaria/semanal.
- --log aplica log al valor (útil para commodities).
"""

def sniff_sep(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read(4096)
    try:
        return csv.Sniffer().sniff(s, delimiters=",;|\t").delimiter
    except Exception:
        return ","

def main():
    if len(sys.argv) < 3:
        print("Uso: py _normalize_exog.py INPUT.csv OUTPUT.csv [--log]"); sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    do_log = "--log" in sys.argv[3:]

    sep = sniff_sep(src)
    df = pd.read_csv(src, sep=sep)
    low = {c: c.lower() for c in df.columns}

    date_cand = {"date","fecha","mes","month","period","time","fecha_mes"}
    val_cand  = {"value","valor","price","precio","index","indice","tipo_cambio","ipc","diesel","usd","corn","soy"}

    date_col = next((c for c in df.columns if low[c] in date_cand), None)
    val_col  = next((c for c in df.columns if low[c] in val_cand), None)
    if date_col is None or val_col is None:
        print("Columnas no encontradas. Encabezados:", list(df.columns)); sys.exit(1)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col]).sort_values(date_col)

    s = pd.Series(df[val_col].values, index=df[date_col]).resample("MS").mean().interpolate(limit_direction="both")
    if do_log:
        s = np.log(s.clip(lower=1e-9))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": s.index, "value": s.values}).to_csv(out, index=False)
    print("OK ->", out)

if __name__ == "__main__":
    main()
