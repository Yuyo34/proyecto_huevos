from __future__ import annotations
import numpy as np
import pandas as pd

def fit_weights(y_true: pd.Series, preds: dict[str, pd.Series]) -> pd.Series:
    df = pd.concat(preds, axis=1)
    y = y_true.loc[df.index]
    denom = (np.abs(y) + np.abs(df).mean(axis=1)) / 2.0
    R = (y.values.reshape(-1,1) - df.values) / np.where(denom.values.reshape(-1,1)==0, 1.0, denom.values.reshape(-1,1))
    k = df.shape[1]
    w = np.ones(k) / k
    lr = 0.5
    for _ in range(200):
        grad = 2 * (R.T @ (R @ w))
        w -= lr * grad
        w = np.maximum(w, 0)
        s = w.sum()
        if s == 0:
            w = np.ones(k)/k
        else:
            w = w / s
    return pd.Series(w, index=df.columns)

def combine(preds: dict[str, pd.Series], weights: pd.Series) -> pd.Series:
    df = pd.concat(preds, axis=1)
    cols = list(weights.index)
    df = df.reindex(columns=cols)
    return (df * weights.values).sum(axis=1)
