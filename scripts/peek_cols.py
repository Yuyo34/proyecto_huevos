import pandas as pd
for p in ["out/fcst_base_h1.csv","out/fcst_base_h2.csv"]:
    df = pd.read_csv(p, nrows=3)
    print("\n==", p, "==")
    print(list(df.columns))
    print(df.head(2).to_string(index=False))
