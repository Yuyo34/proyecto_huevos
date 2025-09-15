import sys, pathlib, csv
import pandas as pd

SRC = "monthly_index.csv"
OUT = pathlib.Path("data") / "precio_huevo_mensual.csv"

def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t").delimiter
    except Exception:
        return ","

sep = sniff_sep(SRC)
df = pd.read_csv(SRC, sep=sep)
df.columns = [c.lower() for c in df.columns]

# Columna de fecha
if "month" not in df.columns:
    print("No hay columna 'month'. Encabezados:", list(df.columns)); sys.exit(1)

# Elegir columna de precio (prioridad: p50_shrunk > p50 > p50_zona_color)
val_col = None
for c in ["p50_shrunk","p50","p50_zona_color"]:
    if c in df.columns:
        val_col = c; break
if val_col is None:
    print("No encontré columna de precio (p50_shrunk/p50/p50_zona_color). Encabezados:", list(df.columns)); sys.exit(1)

# Ponderador
if "n" not in df.columns:
    df["n"] = 1.0
df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0.0)

# Convertir a numérico y fechas
df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.dropna(subset=["month", val_col])

# Promedio ponderado por mes
g = df.groupby(df["month"].dt.to_period("M"))
def wavg(x):
    w = x["n"].sum()
    if w > 0:
        return (x[val_col] * x["n"]).sum() / w
    return x[val_col].mean()
s = g.apply(wavg).to_timestamp(how="start").sort_index()

out = pd.DataFrame({"date": s.index, "value": s.values})
out.to_csv(OUT, index=False, encoding="utf-8")
print("OK ->", OUT)
