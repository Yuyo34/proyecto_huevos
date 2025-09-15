from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.io_tools import read_series_csv
from ..utils.exog_tools import build_exog_matrix
from ..modeling.sarimax_exog import SarimaxExog
from ..eval.backtesting import rolling_backtest
from ..utils.metrics import mape, smape, mase
from ..utils.stl_tools import stl_decompose

BASE_REGRESSORS = ["usdclp", "ipc", "diesel", "corn", "soy"]

def _fallback_forecast(y: pd.Series, h: int, seasonality: int = 12) -> pd.Series:
    y = y.dropna().astype(float)
    if len(y) == 0:
        raise ValueError("Fallback sin datos: la serie está vacía.")
    last_ts = y.index[-1]
    future_idx = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    if len(y) >= seasonality:
        tail = y.iloc[-seasonality:]
        vals = [float(tail.iloc[i % seasonality]) for i in range(h)]
        return pd.Series(vals, index=future_idx)
    if len(y) >= 2:
        slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        vals = y.iloc[-1] + slope * np.arange(1, h+1)
        return pd.Series(vals, index=future_idx)
    return pd.Series([float(y.iloc[-1])] * h, index=future_idx)

def _seasonal_pattern(seasonal: pd.Series) -> pd.Series:
    pat = seasonal.groupby(seasonal.index.month).mean()
    pat.index = range(1, 13)
    return pat

def _build_X(y, exog_paths, lags, log_names):
    exog = {}
    for name, path in exog_paths.items():
        if path:
            exog[name] = read_series_csv(path).asfreq("MS").interpolate(limit_direction="both")
    if not exog:
        return None
    X = build_exog_matrix(y, exog, lags=lags, log_transform=log_names)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    for r in BASE_REGRESSORS:
        ap.add_argument(f"--{r}", required=False)
    ap.add_argument("--lags", nargs="+", type=int, default=[1,2])
    ap.add_argument("--seasonality", type=int, default=12)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--multiplicative", action="store_true")
    ap.add_argument("--min_init", type=int, default=6)
    ap.add_argument("--bt_init", type=int, default=36)
    args = ap.parse_args()

    # 1) Target
    y = read_series_csv(args.target).asfreq("MS")
    if y.isna().any():
        y = y.interpolate(limit_direction="both")
    if args.multiplicative and (y <= 0).any():
        eps = max(y[y > 0].min() * 0.1, 1e-6)
        y = y.clip(lower=eps)

    # 2) STL
    n = len(y.dropna())
    min_for_stl = max(13, args.seasonality * 2 + 1)
    if n >= min_for_stl:
        try:
            if args.multiplicative:
                ylog = np.log(y)
                trend_log, seasonal_log, resid_log = stl_decompose(ylog, period=args.seasonality, robust=True)
                seasonal = np.exp(seasonal_log)
                y_deseas = (y / seasonal).dropna()
                print(f"[STL] multiplicativa OK (n={n}, period={args.seasonality})")
            else:
                trend, seasonal_add, resid = stl_decompose(y, period=args.seasonality, robust=True)
                seasonal = seasonal_add
                y_deseas = (y - seasonal).dropna()
                print(f"[STL] aditiva OK (n={n}, period={args.seasonality})")
        except Exception as e:
            print(f"[STL] Falló ({e}); usando sin desestacionalizar.")
            seasonal = pd.Series(1.0 if args.multiplicative else 0.0, index=y.index)
            y_deseas = y.copy()
    else:
        print(f"[STL] Serie corta (n={n}<{min_for_stl}); sin STL.")
        seasonal = pd.Series(1.0 if args.multiplicative else 0.0, index=y.index)
        y_deseas = y.copy()

    # 3) Exógenas con reintento (lags originales -> si vacías, usar lag0)
    exog_paths = {r: getattr(args, r) for r in BASE_REGRESSORS}
    X = _build_X(y, exog_paths, lags=args.lags, log_names=["usdclp","diesel","corn","soy"])
    def clean_X(X_in, y_index):
        X_al = X_in.loc[y_index].ffill().bfill().dropna(axis=1, how="all")
        return X_al
    X_model = None
    y_model  = y_deseas

    if X is not None and X.shape[1] > 0:
        X_imp = clean_X(X, y_deseas.index)
        if X_imp.shape[1] == 0:
            print(f"[EXOG] 0 columnas con lags={args.lags}. Reintentando con lag0…")
            X0 = _build_X(y, exog_paths, lags=[0], log_names=["usdclp","diesel","corn","soy"])
            if X0 is not None:
                X0_imp = clean_X(X0, y_deseas.index)
                if X0_imp.shape[1] > 0:
                    X_model = X0_imp.dropna()
                    y_model = y_deseas.loc[X_model.index]
                    print(f"[EXOG] OK con lag0: {X_model.shape[1]} features en {len(X_model)} filas.")
                else:
                    print("[EXOG] lag0 también vacío. Uso univariado.")
            else:
                print("[EXOG] No se pudo construir X con lag0. Uso univariado.")
        else:
            # quitar filas con NaN
            X_rows = X_imp.dropna()
            if len(X_rows) == 0:
                print("[EXOG] Sin filas válidas tras imputación (lags originales). Probando lag0…")
                X0 = _build_X(y, exog_paths, lags=[0], log_names=["usdclp","diesel","corn","soy"])
                if X0 is not None:
                    X0_imp = clean_X(X0, y_deseas.index).dropna()
                    if len(X0_imp) > 0:
                        X_model = X0_imp
                        y_model = y_deseas.loc[X_model.index]
                        print(f"[EXOG] OK con lag0: {X_model.shape[1]} features en {len(X_model)} filas.")
                    else:
                        print("[EXOG] lag0 sin filas válidas. Uso univariado.")
                else:
                    print("[EXOG] No se pudo construir X con lag0. Uso univariado.")
            else:
                X_model = X_rows
                y_model = y_deseas.loc[X_model.index]
                print(f"[EXOG] Usando {X_model.shape[1]} features en {len(X_model)} filas.")
    else:
        print("[EXOG] No hay exógenas o quedaron vacías; uso univariado.")

    # 4) Backtesting (métricas en escala original)
    n_d = len(y_model)
    initial_window = max(args.min_init, min(n_d-1, args.bt_init))
    can_backtest = (n_d - args.h) >= initial_window and initial_window >= args.min_init

    def builder():
        return SarimaxExog(seasonal_period=args.seasonality,
                           pdq_grid=[(0,1,1),(1,1,0),(1,1,1)],
                           PDQ_grid=[(0,1,1),(1,1,0)],
                           trend=None)

    def metrics_on_original_scale(preds_deseas: pd.Series) -> dict:
        pat = _seasonal_pattern(seasonal)
        seas_bt = preds_deseas.index.to_series().dt.month.map(pat).astype(float)
        preds_orig = (preds_deseas * seas_bt.values) if args.multiplicative else (preds_deseas + seas_bt.values)
        y_true_orig = y.reindex(preds_orig.index)
        return {"MAPE": mape(y_true_orig, preds_orig),
                "sMAPE": smape(y_true_orig, preds_orig),
                "MASE": mase(y, preds_orig, seasonality=args.seasonality)}

    if can_backtest:
        bt = rolling_backtest(y_model, X_model, builder, horizon=1, initial_window=initial_window, seasonality=args.seasonality)
        if "pred" in bt and not bt["pred"].empty:
            metrics_orig = metrics_on_original_scale(bt["pred"])
            print(f"[BT] Ventana inicial={initial_window}, puntos={n_d} -> métricas (ORIG):", metrics_orig)
        else:
            print(f"[BT] Sin predicciones (n_d={n_d}).")
    else:
        print(f"[BT] Sin backtest (n_d={n_d}, init={initial_window}).")

    # 5) Ajuste final
    last_idx = y.index[-1]
    future_idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=args.h, freq="MS")
    try:
        model = builder()
        model.fit(y_model, X_model)
        if X_model is not None:
            last_row = X_model.iloc[[-1]].to_numpy()
            X_future = pd.DataFrame(np.repeat(last_row, args.h, axis=0), index=future_idx, columns=X_model.columns)
        else:
            X_future = None
        deseas_forecast = model.forecast(steps=args.h, X_future=X_future)
    except Exception as e:
        print(f"[SARIMAX] Falló ({e}); usando fallback baseline.")
        y_for_fb = y_model if len(y_model) > 0 else y_deseas
        deseas_forecast = _fallback_forecast(y_for_fb, args.h, seasonality=args.seasonality)

    # 6) Reaplicar estacionalidad
    pat = _seasonal_pattern(seasonal)
    seas_fut = pd.Series(future_idx.month, index=future_idx).map(pat).astype(float)
    y_forecast = (deseas_forecast * seas_fut.values) if args.multiplicative else (deseas_forecast + seas_fut.values)

    # 7) Guardar
    out = pd.Series(y_forecast, index=future_idx, name="forecast")
    out_df = out.to_frame()
    out_df["date"] = out_df.index
    out_df = out_df[["date","forecast"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Pronóstico guardado en {args.out}")

if __name__ == "__main__":
    main()

