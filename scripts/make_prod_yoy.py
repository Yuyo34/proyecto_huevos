import sys, pandas as pd
src = sys.argv[1] if len(sys.argv)>1 else "data/chilehuevos_produccion.csv"
dst = sys.argv[2] if len(sys.argv)>2 else "data/prod_yoy.csv"
s = pd.read_csv(src, parse_dates=["date"]).sort_values("date").set_index("date")["value"].asfreq("MS")
yoy = (s / s.shift(12) - 1.0) * 100
pd.DataFrame({"date": yoy.index, "value": yoy.values}).dropna().to_csv(dst, index=False)
print(f"OK -> {dst}")
