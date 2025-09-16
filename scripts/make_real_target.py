import json, urllib.request, pandas as pd
from pathlib import Path

t = pd.read_csv("data/precio_huevo_mensual.csv", parse_dates=["date"]).dropna()
t = t.sort_values("date").set_index("date")["value"]

y0, y1 = t.index.min().year, t.index.max().year+1
def fetch_ipc(y):
    with urllib.request.urlopen(f"https://mindicador.cl/api/ipc/{y}") as r:
        data = json.load(r)
    df = pd.DataFrame(data.get("serie", []))
    if df.empty: return pd.DataFrame(columns=["fecha","valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.tz_localize(None)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna(subset=["fecha","valor"])

ipc = pd.concat([fetch_ipc(y) for y in range(y0, y1+1)], ignore_index=True)
ipc = ipc.set_index("fecha")["valor"].resample("MS").mean().sort_index()

idx = (1 + ipc/100.0).cumprod()
idx = (idx/idx.iloc[0])*100
idx = idx.reindex(t.index).interpolate("time").ffill().bfill()

real = t * (100.0/idx)
Path("data").mkdir(exist_ok=True)
pd.DataFrame({"date": real.index, "value": real.values}).to_csv("data/precio_huevo_mensual_real.csv", index=False)
print("OK -> data/precio_huevo_mensual_real.csv")
