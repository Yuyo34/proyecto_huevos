import json, urllib.request, pandas as pd
from datetime import date
from pathlib import Path

def fetch_year(code, year):
    url = f"https://mindicador.cl/api/{code}/{year}"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    df = pd.DataFrame(data.get("serie", []))
    if df.empty:
        return pd.DataFrame(columns=["fecha","valor"])
    # Fechas a tz-aware y luego a naive (sin tz)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", utc=True).dt.tz_convert(None)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df[["fecha","valor"]].dropna()

def fetch_range(code, y0, y1):
    dfs = []
    for y in range(y0, y1+1):
        try:
            dfs.append(fetch_year(code, y))
        except Exception:
            pass
    if not dfs:
        return pd.Series(dtype=float)
    df = pd.concat(dfs, ignore_index=True).dropna().sort_values("fecha")
    s = pd.Series(df["valor"].values, index=df["fecha"])
    # Pasar a mensual (MS) promediando si es diario
    s = s.resample("MS").mean().interpolate("time")
    return s

# Rango según tu target
t = pd.read_csv("data/precio_huevo_mensual.csv", parse_dates=["date"])
t = t.dropna(subset=["date","value"]).sort_values("date")
y0 = int(t["date"].min().year)
y1 = int(date.today().year)

usd = fetch_range("dolar", y0, y1)
ipc = fetch_range("ipc",   y0, y1)

Path("data").mkdir(exist_ok=True)
pd.DataFrame({"date": usd.index, "value": usd.values}).to_csv("data/usdclp.csv", index=False)
pd.DataFrame({"date": ipc.index, "value": ipc.values}).to_csv("data/ipc.csv", index=False)
print(f"OK -> data/usdclp.csv [{usd.index.min().date()} .. {usd.index.max().date()}] ({len(usd)} filas)")
print(f"OK -> data/ipc.csv   [{ipc.index.min().date()} .. {ipc.index.max().date()}] ({len(ipc)} filas)")
