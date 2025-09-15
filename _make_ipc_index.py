import pandas as pd
df = pd.read_csv("data/ipc.csv", parse_dates=["date"]).dropna().sort_values("date")
level = 100.0
rows=[]
for _,r in df.iterrows():
    level *= (1+float(r["value"])/100.0)
    rows.append({"date": r["date"], "value": level})
pd.DataFrame(rows).to_csv("data/ipc_index.csv", index=False)
print("OK -> data/ipc_index.csv")
