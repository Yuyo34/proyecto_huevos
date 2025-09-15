from __future__ import annotations
import pandas as pd
from statsmodels.tsa.seasonal import STL

def stl_decompose(series: pd.Series, period: int = 12, robust: bool = True):
    series = series.asfreq(series.index.inferred_freq) if series.index.inferred_freq else series
    stl = STL(series, period=period, robust=robust)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid

def deseasonalize(series: pd.Series, seasonal: pd.Series, multiplicative: bool = True) -> pd.Series:
    aligned = series.align(seasonal, join="inner")[0]
    seasonal = seasonal.loc[aligned.index]
    if multiplicative:
        return aligned / seasonal.replace(0, pd.NA)
    return aligned - seasonal

def reseasonalize(pred: pd.Series, seasonal: pd.Series, multiplicative: bool = True) -> pd.Series:
    seasonal = seasonal.loc[pred.index.intersection(seasonal.index)]
    if multiplicative:
        return pred * seasonal
    return pred + seasonal
