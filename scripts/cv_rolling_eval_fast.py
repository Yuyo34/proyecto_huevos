import pandas as pd, numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET = "data/precio_huevo_mensual_real_FOR_ROCV.csv"
FREQ   = "MS"
SEAS   = 12
BT_INIT= 24

def read_series_csv(path):
    df   = pd.read_csv(path)
    dcol = next(c for c in df.columns if c.lower() in ("date","ds","fecha"))
    vcol = next(c for c in df.columns if c.lower() in ("value","y","target","actual","real"))
    idx  = (pd.to_datetime(df[dcol], errors="coerce")
              .dt.to_period("M")
              .dt.to_timestamp(how="start"))
    s    = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values, index=idx).sort_index()
    # quita duplicados de mes y fuerza frecuencia mensual
    s    = s[~s.index.duplicated(keep="last")].asfreq(FREQ)
    return s

def fit_predict(y_tr: pd.Series, h: int) -> float:
    y_tr = y_tr.astype("float64")
    order      = (1,0,0)
    seas_order = (0,1,1,SEAS)
    m   = SARIMAX(y_tr, order=order, seasonal_order=seas_order,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = m.fit(disp=False)
    fc  = res.get_forecast(steps=h).predicted_mean
    return float(fc.iloc[-1])

def run(h: int, out_path: str):
    y = read_series_csv(TARGET).dropna()
    print(f"Len(y)={len(y)}; rango= {y.index.min().date()} -> {y.index.max().date()}")
    folds = []
    idx_full = pd.date_range(y.index.min(), y.index.max(), freq=FREQ)

    # i recorre finales de ventana de entrenamiento
    starts = range(BT_INIT, len(idx_full)-h)
    total  = len(starts)
    for k, i in enumerate(starts, start=1):
        t_end = idx_full[i-1]
        y_tr  = y.loc[:t_end].dropna()
        if len(y_tr) < max(BT_INIT, SEAS+2):
            print(f"[h={h}] fold {k}/{total} SKIP len={len(y_tr)}")
            continue
        try:
            yhat = fit_predict(y_tr, h)
            folds.append({
                "date": t_end + pd.offsets.MonthBegin(h),
                "y_hat": yhat,
                "split_end": t_end
            })
            print(f"[h={h}] fold {k}/{total} train<= {t_end:%Y-%m} OK")
        except Exception as e:
            print(f"[h={h}] fold {k}/{total} X {e}")

    if not folds:
        print(f"[h={h}] sin folds válidos; no se genera {out_path}")
        return

    df = pd.DataFrame(folds).sort_values("date")
    df.to_csv(out_path, index=False)
    print(f"OK -> {out_path} ({len(df)} filas)")

if __name__ == "__main__":
    run(1, "out/rocv_h1_pred.csv")
    run(2, "out/rocv_h2_pred.csv")

