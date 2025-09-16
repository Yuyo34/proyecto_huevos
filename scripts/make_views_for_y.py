import pandas as pd, numpy as np
from pathlib import Path

Path("data").mkdir(exist_ok=True)

def load(path):
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].asfreq("MS")

y   = load("data/precio_huevo_mensual_real.csv").dropna()   # ancla de calendario
usd = load("data/usdclp_dlog.csv")
ipc = load("data/imacec_yoy.csv")
soy = load("data/prod_yoy.csv")

# recortar exógenas al índice de y (sin inventar datos)
usd_y = usd.reindex(y.index)
ipc_y = ipc.reindex(y.index)
soy_y = soy.reindex(y.index)

# tirar filas donde falte y o alguna exógena clave
df = pd.concat({"y":y,"usd":usd_y,"ipc":ipc_y,"soy":soy_y}, axis=1).dropna()

def save(name, s):
    pd.DataFrame({"date": s.index, "value": s.values}).to_csv(f"data/{name}.csv", index=False)

save("precio_huevo_mensual_real_for_y", df["y"])
save("usdclp_dlog_for_y",              df["usd"])
save("imacec_yoy_for_y",               df["ipc"])
save("prod_yoy_for_y",                 df["soy"])

# variantes de producción sobre el mismo índice:
# lead1/lead2 (oferta adelantada) y suavizado MA3
for k in (1,2):
    lead = df["soy"].shift(-k).dropna()
    pd.DataFrame({"date": lead.index, "value": lead.values}).to_csv(f"data/prod_yoy_for_y_lead{k}.csv", index=False)

ma3 = df["soy"].rolling(3, min_periods=1).mean()
pd.DataFrame({"date": ma3.index, "value": ma3.values}).to_csv("data/prod_yoy_for_y_ma3.csv", index=False)

# dlog mensual de producción (mejor para corto plazo en muchos casos)
prod_lvl = load("data/chilehuevos_produccion.csv").reindex(y.index)
prod_lvl = prod_lvl.replace([np.inf,-np.inf], np.nan)
prod_lvl = prod_lvl.where(prod_lvl>0)
dlog = (np.log(prod_lvl) - np.log(prod_lvl.shift(1)))*100
dlog = dlog.reindex(df.index).dropna()
pd.DataFrame({"date": dlog.index, "value": dlog.values}).to_csv("data/prod_dlog_for_y.csv", index=False)

print("Ventana final for_y:", df.index.min(), "→", df.index.max(), f"({len(df)} meses)")
