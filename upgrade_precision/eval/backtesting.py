from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable, Dict
from ..utils.metrics import mape, smape, mase

def _fallback_forecast(y: pd.Series, h: int, seasonality: int = 12) -> pd.Series:
    y = y.dropna().astype(float)
    last_ts = y.index[-1]
    future_idx = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    # 1) seasonal-naive si tengo al menos un ciclo completo
    if len(y) >= seasonality:
        tail = y.iloc[-seasonality:]
        vals = [float(tail.iloc[i % seasonality]) for i in range(h)]
        return pd.Series(vals, index=future_idx)
    # 2) drift si tengo >=2 puntos
    if len(y) >= 2:
        slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        vals = y.iloc[-1] + slope * np.arange(1, h+1)
        return pd.Series(vals, index=future_idx)
    # 3) último valor
    return pd.Series([float(y.iloc[-1])] * h, index=future_idx)

# model_builder: Callable[[], model]  con .fit(y, X) y .forecast(steps, X_future)
def rolling_backtest(y: pd.Series,
                     X: pd.DataFrame | None,
                     model_builder: Callable,
                     horizon: int = 1,
                     initial_window: int = 24,
                     step: int = 1,
                     seasonality: int = 12) -> Dict:
    y = y.dropna().astype(float)
    preds = []
    for end in range(initial_window, len(y)-horizon+1, step):
        y_train = y.iloc[:end]
        y_test  = y.iloc[end:end+horizon]
        X_train = X.iloc[:end] if X is not None else None
        X_test  = X.iloc[end:end+horizon] if X is not None else None

        # Si la ventana es demasiado corta para un SARIMAX estacional, usa fallback
        if len(y_train) < max(24, 2*seasonality):
            y_hat = _fallback_forecast(y_train, horizon, seasonality)
            preds.append(y_hat)
            continue

        # Intento con SARIMAX; si falla, fallback
        try:
            model = model_builder()
            model.fit(y_train, X_train)
            y_hat = model.forecast(steps=horizon, X_future=X_test)
        except Exception:
            y_hat = _fallback_forecast(y_train, horizon, seasonality)
        preds.append(y_hat)

    if not preds:
        # Serie demasiado corta: devuelve métricas vacías pero sin romper
        return {"pred": pd.Series(dtype=float), "metrics": {"MAPE": float("nan"), "sMAPE": float("nan"), "MASE": float("nan")}}

    preds = pd.concat(preds).sort_index()
    y_test_all = y.loc[preds.index]
    metrics = {
        "MAPE": mape(y_test_all, preds),
        "sMAPE": smape(y_test_all, preds),
        "MASE": mase(y.loc[:y_test_all.index.max()], preds, seasonality=seasonality)
    }
    return {"pred": preds, "metrics": metrics}
