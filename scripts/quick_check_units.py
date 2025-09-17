import pandas as pd
import numpy as np
from pathlib import Path

p = Path("data/precio_huevo_mensual_real.csv")
df = pd.read_csv(p)
# Detecta columnas
tcol = next(c for c in df.columns if c.lower() in ["date","ds","fecha"])
vcol = next(c for c in df.columns if c.lower() in ["value","y","target","actual","y_true"])

idx = pd.to_datetime(df[tcol], errors="coerce").dt.to_period("M").dt.to_timestamp()
s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").to_numpy(), index=idx).dropna().sort_index()

last12 = s.iloc[-12:]
print("Unidades = las del TARGET (mismas que el forecast)")
print("Últimos 12 meses del TARGET (CLP reales):")
print(f"  min={last12.min():,.2f}  mediana={last12.median():,.2f}  max={last12.max():,.2f}")
print(f"Último valor observado: {s.iloc[-1]:,.2f} CLP")
