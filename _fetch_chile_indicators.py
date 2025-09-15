import json, urllib.request, pandas as pd
from pathlib import Path

def fetch_indicator(code):
    url = f"https://mindicador.cl/api/{code}"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    df = pd.DataFrame(data["serie"])
    # columnas esperadas: fecha, valor
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha","valor"]).sort_values("fecha")
    s = pd.Series(df["valor"].values, index=df["fecha"]).resample("MS").mean()
    return pd.DataFrame({"date": s.index, "value": s.values})

Path("data").mkdir(exist_ok=True)
fetch_indicator("dolar").to_csv("data/usdclp.csv", index=False)  # USD/CLP
fetch_indicator("ipc").to_csv("data/ipc.csv", index=False)       # IPC mensual

print("OK -> data/usdclp.csv, data/ipc.csv")
