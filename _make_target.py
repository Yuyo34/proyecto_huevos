import sys, pathlib, csv
import pandas as pd

SRC = "monthly_index.csv"
OUT = pathlib.Path("data") / "precio_huevo_mensual.csv"

# 1) Autodetectar separador
def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t").delimiter
    except Exception:
        return ","

# 2) Leer y detectar columnas de fecha/valor
sep = sniff_sep(SRC)
df = pd.read_csv(SRC, sep=sep)

lower = {c: c.lower() for c in df.columns}
date_candidates = {"date","fecha","period","mes","month"}
val_candidates  = {"value","precio","price","valor","index","indice"}

date_col = next((c for c in df.columns if lower[c] in date_candidates), None)
val_col  = next((c for c in df.columns if lower[c] in val_candidates), None)

if date_col is None or val_col is None:
    print("No se encontraron columnas de fecha/valor en monthly_index.csv. Encabezados:", list(df.columns))
    sys.exit(1)

# 3) Normalizar fechas y ordenar
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)

# 4) Guardar como date,value (frecuencia mensual esperada por el pipeline)
out = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "value"})
out.to_csv(OUT, index=False, encoding="utf-8")
print("OK ->", OUT)
