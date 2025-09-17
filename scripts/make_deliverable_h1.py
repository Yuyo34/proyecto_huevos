import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
BASE = Path(".")
# Archivos de entrada (ajusta si usas otros nombres)
FCST = Path("out/fcst_base_h1.csv") if Path("out/fcst_exog_h1.csv").exists()==False else Path("out/fcst_exog_h1.csv")
TARGET = Path("data/precio_huevo_mensual_real.csv")

def pick(df, kind):
    m = dict(
        date=["date","ds","fecha"],
        pred=["forecast","y_hat","yhat","pred","prediction","y_pred","fcst","hat"],
        val =["value","y","target","actual","y_true"]
    )[kind]
    for c in df.columns:
        if c.lower() in m: return c
    return None

def to_month_index(s):
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

# --- carga forecast h=1 ---
if not FCST.exists():
    raise SystemExit(f"No encuentro {FCST}. Corre el pipeline h=1 primero.")
dfp = pd.read_csv(FCST)
dcol = pick(dfp,"date") or dfp.columns[0]
pcol = pick(dfp,"pred");  assert pcol, f"Columnas forecast no detectadas en {FCST}: {list(dfp.columns)}"
f = pd.Series(pd.to_numeric(dfp[pcol], errors="coerce").to_numpy(), index=to_month_index(dfp[dcol]))
f = f[~f.index.duplicated(keep="last")].sort_index()

# --- carga target para graficar últimos 24 meses ---
y = None
if TARGET.exists():
    dft = pd.read_csv(TARGET)
    t_d = pick(dft,"date") or dft.columns[0]
    t_v = pick(dft,"val")  or dft.columns[1]
    y = pd.Series(pd.to_numeric(dft[t_v], errors="coerce").to_numpy(), index=to_month_index(dft[t_d]))
    y = y[~y.index.duplicated(keep="last")].sort_index()

# Último pronóstico disponible (normalmente el próximo mes tras tu último dato observado)
last_date = f.index.max()
last_value = float(f.loc[last_date])

# Carpeta de salida con timestamp
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
outdir = BASE / "deliverables" / ts
outdir.mkdir(parents=True, exist_ok=True)

# 1) CSV resumen
summary = pd.DataFrame([{"horizon":"h=1 (1 mes)", "periodo": last_date.date().isoformat(), "precio_pronosticado": round(last_value,2)}])
summary.to_csv(outdir/"forecast_h1_summary.csv", index=False)

# 2) Gráfico (últimos 24m de y si existe + f superpuesto)
plt.figure(figsize=(9,4.8))
if y is not None and len(y)>0:
    yy = y.iloc[-24:]
    plt.plot(yy.index, yy.values, label="Observado")
plt.plot(f.index, f.values, linestyle="--", marker="o", label="Pronóstico h=1")
plt.title("Precio del huevo - Observado vs Pronóstico h=1")
plt.xlabel("Mes"); plt.ylabel("Precio (CLP, real)")
plt.legend(); plt.tight_layout()
plt.savefig(outdir/"forecast_h1_chart.png", dpi=160)
plt.close()

# 3) Resumen de calidad (si existe out/rocv_quality.csv lo incluye)
quality_path = Path("out/rocv_quality.csv")
if quality_path.exists():
    q = pd.read_csv(quality_path)
    q.to_csv(outdir/"quality_snapshot.csv", index=False)

# Imprime para consola (para tu email/whatsapp)
print("=== RESUMEN PRONÓSTICO h=1 ===")
print(f"Periodo pronosticado: {last_date:%Y-%m}")
print(f"Precio estimado     : {last_value:,.2f} CLP")
print(f"Archivos: {outdir/'forecast_h1_summary.csv'}, {outdir/'forecast_h1_chart.png'}")
if quality_path.exists():
    print(f"Calidad (snapshot)  : {outdir/'quality_snapshot.csv'}")
