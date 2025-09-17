import os, pandas as pd, numpy as np

FORECAST_PATHS = ["out/forecast_h1.csv","out/fcst_base_h1.csv"]
CPI_CANDIDATES = ["data/ipc_index.csv","data/ipc_general.csv","data/ine_ipc.csv","data/ipc.csv"]

# Asunción por defecto si no hay IPC de nivel: inflación mensual (MoM) en %
MOM_ASSUMPTION_PCT = 0.5  # <- cámbialo si quieres

def find_first(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

def read_date_value_csv(path):
    df = pd.read_csv(path)
    dcol = next(c for c in df.columns if c.lower() in ["date","ds","fecha"])
    vcol = next(c for c in df.columns if c.lower() in ["value","y","target","actual","y_true","forecast","yhat","pred","prediction","y_pred","fcst","hat"])
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.sort_values(dcol).dropna(subset=[dcol, vcol])
    return df, dcol, vcol

# 1) Carga forecast real (h=1)
fpath = find_first(FORECAST_PATHS)
if not fpath:
    raise SystemExit("No encontré forecast h=1 (out/forecast_h1.csv ni out/fcst_base_h1.csv).")

fc, dcol_f, pcol_f = read_date_value_csv(fpath)
next_date = pd.to_datetime(fc[dcol_f].iloc[-1])
yhat_real = float(pd.to_numeric(fc[pcol_f], errors="coerce").iloc[-1])

# 2) Carga TARGET real para ubicar el último mes observado (base de deflactación típica)
tgt, dcol_t, vcol_t = read_date_value_csv("data/precio_huevo_mensual_real.csv")
last_row = tgt.iloc[-1]
last_date = pd.to_datetime(last_row[dcol_t])
last_real = float(pd.to_numeric(last_row[vcol_t], errors="coerce"))

# 3) Intenta usar IPC de nivel; si no hay, usa MoM asumido
cpi_path = find_first(CPI_CANDIDATES)
notes = []
if cpi_path:
    cpi, dcol_c, vcol_c = read_date_value_csv(cpi_path)
    # Nos aseguramos frecuencia MS y tomamos el índice como nivel (descarta yoy/mom si viniera así)
    cpi = cpi.rename(columns={dcol_c:"date", vcol_c:"ipc_val"}).set_index("date").sort_index()
    # Heurística rápida: ¿niveles típicos (ej. 80-200)? Si no, caemos a asunción.
    med = float(cpi["ipc_val"].tail(12).median())
    if 50 <= med <= 300:
        # Usamos nivel IPC base en último mes observado; si falta el mes proyectado, extrapolamos por MoM última media(3)
        cpi = cpi.asfreq("MS")
        if next_date not in cpi.index:
            last_moms = cpi["ipc_val"].pct_change().dropna().tail(3)
            mom = last_moms.mean() if len(last_moms)>0 else MOM_ASSUMPTION_PCT/100.0
            steps = ((next_date.year - cpi.index[-1].year)*12 + (next_date.month - cpi.index[-1].month))
            ipc_next = cpi["ipc_val"].iloc[-1] * ((1+mom)**steps)
            cpi.loc[next_date, "ipc_val"] = ipc_next
        ipc_base = float(cpi.loc[last_date, "ipc_val"])
        ipc_next = float(cpi.loc[next_date, "ipc_val"])
        infl_factor = ipc_next / ipc_base if ipc_base>0 else np.nan
        notes.append(f"IPC usado: {os.path.basename(cpi_path)}; factor={infl_factor:.4f}")
    else:
        infl_factor = (1 + MOM_ASSUMPTION_PCT/100.0) ** ( (next_date.year-last_date.year)*12 + (next_date.month-last_date.month) )
        notes.append(f"IPC no en nivel reconocible; usé MoM asumido={MOM_ASSUMPTION_PCT:.2f}%")
else:
    infl_factor = (1 + MOM_ASSUMPTION_PCT/100.0) ** ( (next_date.year-last_date.year)*12 + (next_date.month-last_date.month) )
    notes.append(f"Sin archivo IPC; usé MoM asumido={MOM_ASSUMPTION_PCT:.2f}%")

yhat_nominal = yhat_real * infl_factor
delta_vs_last = 100*(yhat_nominal/last_real - 1.0)

ent = pd.DataFrame([{
    "fecha": next_date.date().isoformat(),
    "precio_nominal_clp": round(yhat_nominal, 2),
    "unidad": "CLP nominales por DOCENA (wholesale)  <-- ajusta el texto si aplica distinto",
    "horizonte_meses": 1,
    "inflacion_factor_utilizado": round(infl_factor, 4),
    "vs_ultimo_obs_real_%": round(delta_vs_last, 1),
    "notas": "; ".join(notes) if notes else ""
}])

print(ent.to_string(index=False))
outp = "out/entregable_proyeccion_h1_nominal.csv"
ent.to_csv(outp, index=False)
print(f"\nOK -> {outp}")
