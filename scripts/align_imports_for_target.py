import pandas as pd

# Índice temporal del target ORIGINAL (no for_y)
ycal = pd.read_csv("data/precio_huevo_mensual_real.csv", parse_dates=["date"]).set_index("date").index

# Lee imports_dlog_for_y.csv tolerante: soporta "date" o primera columna sin nombre
df = pd.read_csv("data/imports_dlog_for_y.csv")
if "date" in df.columns:
    dt = pd.to_datetime(df["date"], errors="coerce")
else:
    first_col = df.columns[0]
    dt = pd.to_datetime(df[first_col], errors="coerce")

valcol = "value" if "value" in df.columns else df.columns[-1]
vals = pd.to_numeric(df[valcol], errors="coerce")

x = pd.Series(vals.values, index=dt).dropna().sort_index()

# Reindexamos al calendario del target, sin asfreq
x0 = x.reindex(ycal)
x0.dropna().to_frame("value").to_csv("data/imports_dlog_FOR_TARGET.csv", index=False)

x1 = x0.shift(-1).dropna()
x1.to_frame("value").to_csv("data/imports_dlog_FOR_TARGET_lead1.csv", index=False)

print("OK -> imports_dlog_FOR_TARGET (+lead1)")
