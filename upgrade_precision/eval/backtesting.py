from __future__ import annotations
import pandas as pd
from typing import Callable, Dict
from .ensemble_weights import fit_weights
from ..utils.metrics import mape, smape, mase

def rolling_backtest(y: pd.Series,
                     X: pd.DataFrame | None,
                     model_builder: Callable,
                     horizon: int = 1,
                     initial_window: int = 24,
                     step: int = 1,
                     seasonality: int = 12) -> Dict:
    y = y.dropna().astype(float)
    preds = []
    test_idx = []
    for end in range(initial_window, len(y)-horizon+1, step):
        y_train = y.iloc[:end]
        y_test = y.iloc[end:end+horizon]
        X_train = X.iloc[:end] if X is not None else None
        X_test = X.iloc[end:end+horizon] if X is not None else None
        model = model_builder()
        model.fit(y_train, X_train)
        y_hat = model.forecast(steps=horizon, X_future=X_test)
        preds.append(y_hat)
        test_idx.extend(y_test.index)
    preds = pd.concat(preds).sort_index()
    y_test_all = y.loc[preds.index]
    metrics = {
        "MAPE": mape(y_test_all, preds),
        "sMAPE": smape(y_test_all, preds),
        "MASE": mase(y.loc[:y_test_all.index.max()], preds, seasonality=seasonality)
    }
    return {"pred": preds, "metrics": metrics}
