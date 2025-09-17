import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

# ---- Config mínimas (ajusta si usas otros nombres) --------------------------
FCST_FILE = r"out\fcst_base_h1.csv"   # forecast real (date, forecast)
TARGET_REAL = r"data\precio_huevo_mensual_real.csv"  # serie real (para contexto gráfico)
IPC_CANDIDATES = [
    r"data\ipc_index.csv", r"data\ipc.csv", r"data\IPC.csv", r"data\ipc_level.csv"
]
OUT_CSV = r"out\entregable_nominal.csv"
OUT_PNG = r"out\entregable_nominal.png"
HIST_MONTHS = 12  # meses de histórico en el gráfico
# -----------------------------------------------------------------------------

def find_date_col(df):
    for c in df.columns:
        if c.lower() in ("date","fecha","ds"):
            return c
    # fallback: primera columna
    return df.columns[0]

def find_value_col(df, prefer=("forecast","yhat","y_hat","pred","prediction","value","valor","price","target","y")):
    # elige la primera columna que suene a valor (distinta de date)
    date_col = find_date_col(df)
    for name in prefer:
        for c in df.columns:
            if c.lower()==name and c!=date_col:
                return c
    # si no, la primera numérica que no sea fecha
    for c in df.columns:
        if c!=date_col and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # última carta
    return [c for c in df.columns if c!=date_col][0]

def read_series_csv(path):
    df = pd.read_csv(path)
    dcol = find_date_col(df)
    vcol = find_value_col(df)
    idx  = pd.to_datetime(df[dcol], errors="coerce").dt.to_period("M").dt.to_timestamp()
    s    = pd.to_numeric(df[vcol], errors="coerce")
    out  = pd.Series(s.values, index=idx).sort_index()
    out.name = vcol
    return out.dropna()

def read_ipc():
    for p in IPC_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            dcol = find_date_col(df)
            # intenta encontrar la columna de índice
            value_cols = [c for c in df.columns if c!=dcol]
            if not value_cols:
                continue
            # elige la primera numérica
            for c in value_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    icol = c; break
            else:
                # fuerza a numérico si vienen strings
                icol = value_cols[0]
                df[icol] = pd.to_numeric(df[icol], errors="coerce")
            idx = pd.to_datetime(df[dcol], errors="coerce").dt.to_period("M").dt.to_timestamp()
            s = pd.Series(df[icol].values, index=idx).sort_index().dropna()
            s.name = "IPC"
            if len(s):
                return s
    return None

def main():
    os.makedirs("out", exist_ok=True)

    # 1) Forecast REAL
    if not os.path.exists(FCST_FILE):
        raise SystemExit(f"No encuentro {FCST_FILE}. Genera el forecast real primero.")
    yhat_real = read_series_csv(FCST_FILE)

    # 2) Target REAL (solo para contexto del gráfico)
    y_real = read_series_csv(TARGET_REAL) if os.path.exists(TARGET_REAL) else None

    # 3) IPC (para pasar de real -> nominal)
    ipc = read_ipc()
    if ipc is None or ipc.empty:
        # Sin IPC: reporta en real y avisa
        fcst_date = yhat_real.index[-1]
        fcst_nominal = yhat_real.iloc[-1]
        print("[AVISO] No encontré archivo de IPC (busqué: " + ", ".join(IPC_CANDIDATES) + ").")
        print("         Reporto el forecast en CLP reales (sin convertir a nominal).")
        print(f"Fecha proyección: {fcst_date:%Y-%m}")
        print(f"Precio estimado (reales): {fcst_nominal:,.2f} CLP por docena")
        # Gráfico simple en reales
        plt.figure(figsize=(8,4.5))
        if y_real is not None:
            y_context = y_real.iloc[-HIST_MONTHS:]
            plt.plot(y_context.index, y_context.values, label="Histórico (reales)")
        plt.scatter([fcst_date], [yhat_real.iloc[-1]], marker="o", label="Forecast (reales)")
        plt.title("Precio mayorista por docena (CLP reales)")
        plt.xlabel("Fecha"); plt.ylabel("CLP")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        # CSV
        pd.DataFrame({
            "date":[fcst_date], "forecast_real":[yhat_real.iloc[-1]],
            "forecast_nominal":[np.nan], "ipc_used":[np.nan],
            "assumption":["sin_ipc"]
        }).to_csv(OUT_CSV, index=False)
        print(f"Guardado: {OUT_CSV} y {OUT_PNG}")
        return

    # 4) Construye factor nominalizador: base = último mes común (y_real vs IPC si hay; si no, usa IPC último)
    if y_real is not None:
        common = sorted(set(y_real.index).intersection(set(ipc.index)))
        base_month = common[-1] if common else ipc.index[-1]
    else:
        base_month = ipc.index[-1]
    base_ipc = float(ipc.loc[base_month])

    # factor(t) = IPC(t) / IPC(base)
    factor = (ipc / base_ipc).sort_index()

    # 5) Convierte forecast real -> nominal (usa IPC del mes t; si falta, último IPC conocido)
    fcst_rows = []
    for t, v in yhat_real.items():
        if t in factor.index:
            f = float(factor.loc[t]); used = "ipc_t"
        else:
            f = float(factor.iloc[-1]); used = "ipc_last"
        fcst_rows.append((t, float(v), f, float(v)*f, used))
    df_fcst = pd.DataFrame(fcst_rows, columns=["date","forecast_real","factor","forecast_nominal","assumption"]).sort_values("date")

    # 6) Imprime en consola el último valor (el que “entregarías”)
    last = df_fcst.iloc[-1]
    print(f"Fecha proyección: {last['date']:%Y-%m}")
    if last["assumption"]=="ipc_last":
        print("Nota: no hay IPC para ese mes; se usó el último IPC disponible.")
    print(f"Precio estimado nominal por docena: {last['forecast_nominal']:,.2f} CLP")

    # 7) Gráfico (histórico nominal + forecast nominal)
    plt.figure(figsize=(9,5))
    if y_real is not None:
        # histórico nominal = real * factor (ajusta al índice disponible)
        f_hist = factor.reindex(y_real.index).fillna(method="ffill")
        y_nom = (y_real * f_hist).dropna()
        y_nom = y_nom.iloc[-HIST_MONTHS:] if len(y_nom)>HIST_MONTHS else y_nom
        plt.plot(y_nom.index, y_nom.values, label="Histórico (nominal)")

    plt.scatter(df_fcst["date"], df_fcst["forecast_nominal"], marker="o", label="Forecast (nominal)")
    # anota el último punto
    plt.annotate(f"{last['forecast_nominal']:,.0f} CLP",
                 xy=(last["date"], last["forecast_nominal"]),
                 xytext=(10, 10), textcoords="offset points")
    plt.title("Precio mayorista por docena (CLP nominales)")
    plt.xlabel("Fecha"); plt.ylabel("CLP")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)

    # 8) CSV de entregable
    df_fcst.to_csv(OUT_CSV, index=False)

    print(f"Guardado: {OUT_CSV} y {OUT_PNG}")
    print(f"Base IPC usada: {base_month:%Y-%m} (IPC={base_ipc})")
    if y_real is not None:
        try:
            ult = float((y_real * factor.reindex(y_real.index).fillna(method='ffill')).dropna().iloc[-1])
            print(f"Último observado (nominal aprox.): {ult:,.2f} CLP")
        except Exception:
            pass

if __name__ == "__main__":
    main()
