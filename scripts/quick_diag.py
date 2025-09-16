import pandas as pd

# Carga y alinea mensual (MS)
y   = pd.read_csv("data/precio_huevo_mensual_real.csv", parse_dates=["date"]).set_index("date")["value"].asfreq("MS")
usd = pd.read_csv("data/usdclp_dlog.csv",             parse_dates=["date"]).set_index("date")["value"].asfreq("MS")
ipc = pd.read_csv("data/imacec_yoy.csv",              parse_dates=["date"]).set_index("date")["value"].asfreq("MS")
soy = pd.read_csv("data/prod_yoy.csv",                parse_dates=["date"]).set_index("date")["value"].asfreq("MS")

df = pd.concat({"y": y, "usd": usd, "ipc": ipc, "soy": soy}, axis=1)

print("Últimas filas:\n", df.tail(6), "\n")
print("NaN por columna:\n", df.isna().sum(), "\n")

# Correlaciones y~soy con lags -6..+6 (lag negativo => oferta ADELANTADA)
cors = []
for k in range(-6, 7):
    c = df["y"].corr(df["soy"].shift(k))
    if pd.notna(c):
        cors.append((k, c))
cors = sorted(cors, key=lambda t: abs(t[1]), reverse=True)[:8]

print("Top correlaciones y ~ soy(lag):")
for k, c in cors:
    print(f"  lag {k:+} meses -> corr={c:.3f}")

# Sugerencia automática simple
if cors:
    best_k, best_c = cors[0]
    if best_k < 0:
        print(f"\nSUGERENCIA: usa oferta adelantada {abs(best_k)}m (lead{abs(best_k)}).")
    elif best_k > 0:
        print(f"\nSUGERENCIA: la oferta rezaga {best_k}m; prueba lags={best_k}.")
    else:
        print("\nSUGERENCIA: relación contemporánea; prueba suavizar (MA3).")
