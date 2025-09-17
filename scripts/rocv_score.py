import os, numpy as np, pandas as pd

TARGET = "data/precio_huevo_mensual_real_FOR_ROCV.csv"
PRED_FILES = ["out/rocv_h1_pred.csv", "out/rocv_h2_pred.csv"]

def _pick(df, kind):
    keys = dict(
        date=["date","ds","fecha"],
        value=["value","y","target","actual","y_true","obs","real","actuals"],
        pred=["y_hat","forecast","yhat","pred","prediction","y_pred","fcst","hat","value_pred"]
    )[kind]
    for c in df.columns:
        if c.lower() in keys: return c
    return None

def _to_month_start(s):
    # Normaliza a inicio de mes; evita el error de freq="MS"
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def read_target(path):
    df = pd.read_csv(path)
    dcol = _pick(df, "date")  or df.columns[0]
    vcol = _pick(df, "value") or df.columns[1]
    idx  = _to_month_start(df[dcol])
    val  = pd.to_numeric(df[vcol], errors="coerce")
    y    = pd.Series(val.to_numpy(), index=idx).dropna()
    y    = y[~y.index.duplicated(keep="last")].sort_index().asfreq("MS")
    return y

def load_pred(path):
    df = pd.read_csv(path)
    dcol = _pick(df, "date") or df.columns[0]
    pcol = _pick(df, "pred")
    if pcol is None:
        raise SystemExit(f"No encuentro columna de predicción en {path}. Cols={list(df.columns)}")
    idx  = _to_month_start(df[dcol])
    val  = pd.to_numeric(df[pcol], errors="coerce")
    yhat = pd.Series(val.to_numpy(), index=idx)
    # quita duplicados por mes conservando la última fila (más reciente)
    yhat = yhat[~yhat.index.duplicated(keep="last")].sort_index().asfreq("MS")
    return yhat, pcol

def score_one(y, pred_path):
    if not os.path.exists(pred_path):
        return dict(file=os.path.basename(pred_path), pred_col=None, rows=0, pairs=0,
                    date_min=None, date_max=None, MAE_model=None, MAE_sNaive=None,
                    rMAE=None, sMAPE=None, MASE=None)
    yhat, pcol = load_pred(pred_path)

    # Meses comunes y con lag t-12 disponible
    common = yhat.index.intersection(y.index)
    pairs  = []
    for t in common:
        tlag = t - pd.offsets.MonthBegin(12)
        if tlag in y.index:
            pairs.append(t)
    pairs = pd.DatetimeIndex(pairs).sort_values()

    rows  = int(len(yhat))
    cnt   = int(len(pairs))
    dmin  = str(pairs.min().date()) if cnt else None
    dmax  = str(pairs.max().date()) if cnt else None

    if cnt == 0:
        return dict(file=os.path.basename(pred_path), pred_col=pcol, rows=rows, pairs=0,
                    date_min=dmin, date_max=dmax, MAE_model=None, MAE_sNaive=None,
                    rMAE=None, sMAPE=None, MASE=None)

    yt   = y.loc[pairs].to_numpy(dtype="float64")
    yph  = yhat.loc[pairs].to_numpy(dtype="float64")
    ylag = y.loc[pairs - pd.offsets.MonthBegin(12)].to_numpy(dtype="float64")

    mae_model  = float(np.mean(np.abs(yt - yph)))
    mae_snaive = float(np.mean(np.abs(yt - ylag)))

    rMAE = mae_model / mae_snaive if mae_snaive > 0 else np.nan
    MASE = rMAE  # MASE out-of-sample consistente con el naïve usado arriba

    denom = (np.abs(yt) + np.abs(yph))
    mask  = denom > 0
    smape = float(np.mean(2.0*np.abs(yt - yph)[mask] / denom[mask]) * 100.0)

    return dict(file=os.path.basename(pred_path), pred_col=pcol, rows=rows, pairs=cnt,
                date_min=dmin, date_max=dmax, MAE_model=round(mae_model,1),
                MAE_sNaive=round(mae_snaive,1), rMAE=round(rMAE,3),
                sMAPE=round(smape,1), MASE=round(MASE,3))

def main():
    y = read_target(TARGET).dropna()
    rows = [score_one(y, p) for p in PRED_FILES]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    os.makedirs("out", exist_ok=True)
    out_path = "out/rocv_quality.csv"
    out.to_csv(out_path, index=False)
    print(f"\nOK -> {out_path}")

if __name__ == "__main__":
    main()
