import pandas as pd, sys
from pathlib import Path

def info(p):
    try:
        df = pd.read_csv(p)
        if not {"date","value"}.issubset(df.columns):
            print(f"{p}: columnas {list(df.columns)} (faltan date/value)")
            return
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date","value"])
        if df.empty:
            print(f"{p}: VACÍO")
            return
        print(f"{p}: {len(df)} filas | min={df['date'].min().date()} max={df['date'].max().date()} sample_head={df.head(2).to_dict('records')}")
    except Exception as e:
        print(f"{p}: error {e}")

for p in ["data/precio_huevo_mensual.csv","data/usdclp.csv","data/ipc.csv"]:
    info(p)
