from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SarimaxExog:
    """SARIMAX con pequeña búsqueda de hiperparámetros por AIC."""
    def __init__(self, seasonal_period: int = 12,
                 pdq_grid = [(0,1,1), (1,1,0), (1,1,1)],
                 PDQ_grid = [(0,1,1), (1,1,0)],
                 trend: str | None = None):
        self.seasonal_period = seasonal_period
        self.pdq_grid = pdq_grid
        self.PDQ_grid = PDQ_grid
        self.trend = trend
        self.best_params_ = None
        self.model_ = None
        self.res_ = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None):
        y = y.astype(float)
        if X is not None:
            X = X.loc[y.index]
        best_aic = np.inf
        best = None
        for pdq in self.pdq_grid:
            for PDQ in self.PDQ_grid:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(y, order=pdq, seasonal_order=(*PDQ, self.seasonal_period),
                                    exog=X, trend=self.trend, enforce_stationarity=False,
                                    enforce_invertibility=False)
                        res = m.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best = (pdq, PDQ, res)
                except Exception:
                    continue
        if best is None:
            raise RuntimeError("No se pudo ajustar ningún modelo SARIMAX válido.")
        self.best_params_ = {"order": best[0], "seasonal_order": (*best[1], self.seasonal_period)}
        self.model_ = best[2].model
        self.res_ = best[2]
        return self

    def _last_and_freq(self):
        """Obtiene último timestamp y frecuencia del modelo de forma robusta."""
        last = None
        freq = None
        # Intento 1: desde model_.data
        try:
            labels = self.model_.data.row_labels
            if hasattr(labels, "__len__") and len(labels) > 0:
                last = labels[-1]
            freq = getattr(self.model_.data, "freq", None)
        except Exception:
            pass
        # Intento 2: desde res_.data
        if last is None:
            try:
                labels = self.res_.data.row_labels
                if hasattr(labels, "__len__") and len(labels) > 0:
                    last = labels[-1]
            except Exception:
                pass
        # Inferir frecuencia si sigue en None
        if freq is None:
            try:
                labels = self.model_.data.row_labels
            except Exception:
                labels = None
            try:
                freq = pd.infer_freq(labels) if labels is not None else None
            except Exception:
                freq = None
        if freq is None:
            # por defecto mensual inicio de mes
            freq = "MS"
        if isinstance(last, pd.Period):
            last = last.to_timestamp()
        return last, freq

    def forecast(self, steps: int = 1, X_future: pd.DataFrame | None = None) -> pd.Series:
        if self.res_ is None:
            raise RuntimeError("Modelo no ajustado.")
        pred = self.res_.get_forecast(steps=steps, exog=X_future)
        mean = pred.predicted_mean
        last, freq = self._last_and_freq()
        # Generar índice futuro genérico (no asumimos MonthBegin explícito)
        future_idx = pd.date_range(start=last, periods=steps+1, freq=freq)[1:]
        mean.index = future_idx
        return mean
