from __future__ import annotations
import numpy as np
import pandas as pd

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100

def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.nanmean(np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)) * 100

def mase(y_true: pd.Series, y_pred: pd.Series, seasonality: int = 12) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    if len(y_true) <= seasonality:
        return np.nan
    naive = y_true.shift(seasonality).iloc[seasonality:]
    errors_model = np.abs(y_true.iloc[seasonality:] - y_pred.iloc[seasonality:])
    errors_naive = np.abs(y_true.iloc[seasonality:] - naive)
    denom = np.nanmean(errors_naive)
    return np.nanmean(errors_model) / denom if denom and not np.isnan(denom) else np.nan

def _align(a: pd.Series, b: pd.Series):
    idx = a.index.intersection(b.index)
    return a.loc[idx].astype(float), b.loc[idx].astype(float)
