import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

def load_series(path):
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].asfreq("MS")
    return s

y   = load_series("data/precio_huevo_mensual_real.csv")
usd = load_series("data/usdclp_dlog.csv")
ipc = load_series("data/imacec_yoy.csv")
soy = load_series("data/prod_yoy.csv")

# Intersección "dura" = meses en que TODAS existen
common_idx = y.dropna().index
for s in (usd, ipc, soy):
    common_idx = common_idx.intersection(s.dropna().index)

# Recorte a la ventana común
y_c   = y.reindex(common_idx)
usd_c = usd.reindex(common_idx)
ipc_c = ipc.reindex(common_idx)
soy_c = soy.reindex(common_idx)

def save(name, s):
    pd.DataFrame({"date": s.index, "value": s.values}).to_csv(f"data/{name}.csv", index=False)

save("precio_huevo_mensual_real_aligned", y_c)
save("usdclp_dlog_aligned", usd_c)
save("imacec_yoy_aligned", ipc_c)
save("prod_yoy_aligned", soy_c)

# Variantes de producción sobre la ventana ya alineada:
# - lead1/lead2 (oferta adelantada)
for k in (1,2):
    lead = soy_c.shift(-k).dropna()
    pd.DataFrame({"date": lead.index, "value": lead.values}).to_csv(f"data/prod_yoy_aligned_lead{k}.csv", index=False)

# - suavizado MA3
ma3 = soy_c.rolling(3, min_periods=1).mean()
pd.DataFrame({"date": ma3.index, "value": ma3.values}).to_csv("data/prod_yoy_aligned_ma3.csv", index=False)

print("Ventana común:", common_idx.min(), "→", common_idx.max(), f"({len(common_idx)} meses)")
print("OK -> data/*_aligned*.csv + lead1/lead2 + ma3")
