import json, urllib.request, pandas as pd, numpy as np
from datetime import date
from pathlib import Path

def fetch_year_mindicador(code, year):
    url = f"https://mindicador.cl/api/{code}/{year}"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    df = pd.DataFrame(data.get("serie", []))
    if df.empty:
        return pd.DataFrame(columns=["fecha","valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df[["fecha","valor"]].dropna()

def to_monthly_mean(s):
    s = s.sort_index()
    # Quitar zona horaria de forma segura (si existe)
    try:
        if getattr(s.index, "tz", None) is not None:
            s = s.tz_convert(None)
    except Exception:
        try:
            s = s.tz_localize(None)
        except Exception:
            pass
    # Promediar a frecuencia mensual (MS) y rellenar suavemente
    s = s.resample("MS").mean().interpolate("time")
    return s

def fetch_range(code, y0, y1):
    dfs = []
    for y in range(y0, y1 + 1):
        try:
            dfs.append(fetch_year_mindicador(code, y))
        except Exception:
            pass
    if not dfs:
        return pd.Series(dtype=float)
    df = pd.concat(dfs, ignore_index=True).dropna().sort_values("fecha")
    s = pd.Series(df["valor"].values, index=df["fecha"])
    return to_monthly_mean(s)

# Rango según el target
t = pd.read_csv("data/precio_huevo_mensual.csv", parse_dates=["date"]).dropna()
t = t.sort_values("date")
y0, y1 = int(t["date"].min().year), int(date.today().year)

usd = fetch_range("dolar",  y0, y1)   # USD/CLP (nivel)
imc = fetch_range("imacec", y0, y1)   # IMACEC (variación mensual en %)

Path("data").mkdir(exist_ok=True)

# USD Δlog mensual
usd = usd.astype(float)
usd = usd[usd > 0]
usd_dlog = np.log(usd).diff().dropna()
pd.DataFrame({"date": usd_dlog.index, "value": usd_dlog.values}).to_csv("data/usdclp_dlog.csv", index=False)

# IMACEC YoY (% interanual) desde variación mensual en %
imc = imc.astype(float)
imc_idx = (1 + (imc / 100.0)).cumprod()
imc_yoy = (imc_idx / imc_idx.shift(12) - 1.0) * 100.0
imc_yoy = imc_yoy.dropna()
pd.DataFrame({"date": imc_yoy.index, "value": imc_yoy.values}).to_csv("data/imacec_yoy.csv", index=False)

print(f"OK -> data/usdclp_dlog.csv [{usd_dlog.index.min().date()}..{usd_dlog.index.max().date()}] ({len(usd_dlog)})")
print(f"OK -> data/imacec_yoy.csv  [{imc_yoy.index.min().date()}..{imc_yoy.index.max().date()}] ({len(imc_yoy)})")
