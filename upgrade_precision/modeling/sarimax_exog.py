from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SarimaxExog:
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

    def forecast(self, steps: int = 1, X_future: pd.DataFrame | None = None) -> pd.Series:
        if self.res_ is None:
            raise RuntimeError("Modelo no ajustado.")
        pred = self.res_.get_forecast(steps=steps, exog=X_future)
        mean = pred.predicted_mean
        mean.index = _next_periods(self.res_.data.row_labels.index[-1], steps, self.model_.data.freq)
        return mean

def _next_periods(last_ts, steps, freq):
    idx = pd.period_range(start=pd.Period(last_ts, freq=freq), periods=steps+1, freq=freq)
    return idx[1:].to_timestamp()
