import pandas as pd
def yoy(src, dst):
    s = pd.read_csv(src, parse_dates=["date"]).set_index("date")["value"].asfreq("MS")
    y = (s/s.shift(12)-1)*100
    pd.DataFrame({"date":y.index,"value":y.values}).dropna().to_csv(dst, index=False)

yoy("data/mercado_pct_importados.csv","data/pct_imp_yoy.csv")

y = pd.read_csv("data/precio_huevo_mensual_real_for_y.csv", parse_dates=["date"]).set_index("date")["value"]
x = pd.read_csv("data/pct_imp_yoy.csv", parse_dates=["date"]).set_index("date")["value"].reindex(y.index)

x.to_frame("value").dropna().to_csv("data/pct_imp_yoy_for_y.csv")
for k in (1,2):
    lead = x.shift(-k).dropna()
    lead.to_frame("value").to_csv(f"data/pct_imp_yoy_for_y_lead{k}.csv")
    print(f"OK -> data/pct_imp_yoy_for_y_lead{k}.csv")
