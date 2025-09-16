import pandas as pd, pathlib
s = pd.read_csv("data/prod_yoy.csv", parse_dates=["date"]).sort_values("date").set_index("date")["value"].asfreq("MS")
for k in (1,2):
    lead = s.shift(-k).dropna()
    pd.DataFrame({"date": lead.index, "value": lead.values}).to_csv(f"data/prod_yoy_lead{k}.csv", index=False)
    print(f"OK -> data/prod_yoy_lead{k}.csv")
