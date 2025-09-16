import pandas as pd
ycal = pd.read_csv("data/precio_huevo_mensual_real.csv", parse_dates=["date"]).set_index("date").index
x = pd.read_csv("data/pct_imp_yoy.csv", parse_dates=["date"]).set_index("date")["value"].asfreq("MS")
x0 = x.reindex(ycal); x0.dropna().to_frame("value").to_csv("data/pct_imp_yoy_FOR_TARGET.csv")
x1 = x0.shift(-1).dropna(); x1.to_frame("value").to_csv("data/pct_imp_yoy_FOR_TARGET_lead1.csv")
x2 = x0.shift(-2).dropna(); x2.to_frame("value").to_csv("data/pct_imp_yoy_FOR_TARGET_lead2.csv")
print("OK -> pct_imp_yoy_FOR_TARGET (+lead1/lead2)")
