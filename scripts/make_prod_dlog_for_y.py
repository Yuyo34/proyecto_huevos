import pandas as pd, numpy as np
s = pd.read_csv("data/chilehuevos_produccion.csv", parse_dates=["date"]).sort_values("date").set_index("date")["value"].asfreq("MS")
s = s.replace([np.inf,-np.inf], np.nan).dropna()
s = s[s>0]
dlog = (np.log(s) - np.log(s.shift(1))) * 100
dlog = dlog.dropna()

# alineamos al calendario del target ya limpio (for_y)
y = pd.read_csv("data/precio_huevo_mensual_real_for_y.csv", parse_dates=["date"]).set_index("date")["value"]
dlog = dlog.reindex(y.index).dropna()
dlog.to_frame("value").to_csv("data/prod_dlog_for_y.csv")
print("OK -> data/prod_dlog_for_y.csv")
