import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p)
if "date" not in df.columns:
    # asume primera es fecha, segunda es valor
    cols = list(df.columns)
    if len(cols) < 2: raise SystemExit("CSV no tiene 2 columnas")
    df = df.rename(columns={cols[0]:"date", cols[1]:"value"})
df.to_csv(p, index=False)
print("Arreglado:", p)
