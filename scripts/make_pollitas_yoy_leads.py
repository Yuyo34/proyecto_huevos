import pandas as pd
from pathlib import Path

p = Path("data/chilehuevos_pollitas.csv")
df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").set_index("date")["value"].asfreq("MS")

yoy = (df / df.shift(12) - 1.0) * 100
yoy = yoy.dropna()
Path("data").mkdir(exist_ok=True)
yoy.to_frame("value").to_csv("data/pollitas_yoy.csv")

for k in (5,6):  # oferta futura
    lead = yoy.shift(-k).dropna()
    lead.to_frame("value").to_csv(f"data/pollitas_yoy_lead{k}.csv")
    print(f"OK -> data/pollitas_yoy_lead{k}.csv")
