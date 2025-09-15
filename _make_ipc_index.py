import pandas as pd
df = pd.read_csv("data/ipc.csv", parse_dates=["date"])
df = df.dropna(subset=["date","value"]).sort_values("date")
# "value" es % mensual (p.ej. 0.6, -0.1, etc.)
level = 100.0
dates, levels = [], []
for _, r in df.iterrows():
    level *= (1 + float(r["value"])/100.0)
    dates.append(r["date"])
    levels.append(level)
out = pd.DataFrame({"date": dates, "value": levels})
out.to_csv("data/ipc_index.csv", index=False)
print("OK -> data/ipc_index.csv (índice acumulado, base 100)")
