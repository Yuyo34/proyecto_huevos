import pandas as pd, numpy as np
from pathlib import Path

src = "data/precio_huevo_mensual_real.csv"
df  = pd.read_csv(src, parse_dates=["date"]).dropna(subset=["date","value"])
df  = df.sort_values("date")

# Serie con índice de fecha
s = pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"]))

# Normaliza a inicio de mes (Period(M) -> Timestamp al inicio del período)
s.index = s.index.to_period("M").to_timestamp()  # <- aquí estaba el bug

# Reindex mensual continuo a MS y rellena huecos cortos (<=2 meses) por tiempo
full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
y_m  = s.reindex(full)
y_m  = y_m.interpolate("time", limit=2, limit_direction="both")

pairs = int((~y_m.isna() & ~y_m.shift(12).isna()).sum())

Path("data").mkdir(exist_ok=True)
pd.DataFrame({"date": y_m.index, "value": y_m.values}).to_csv(
    "data/precio_huevo_mensual_real_FOR_ROCV.csv", index=False
)
print("OK -> data/precio_huevo_mensual_real_FOR_ROCV.csv",
      "rows=", int(y_m.notna().sum()),
      "t-12 pairs=", pairs)
