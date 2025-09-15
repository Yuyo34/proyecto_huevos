from __future__ import annotations
import pandas as pd

def read_series_csv(path: str, date_col: str = "date", value_col: str = "value",
                    freq: str = "MS") -> pd.Series:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    s = pd.Series(df[value_col].values, index=df[date_col])
    s = s.asfreq(freq)
    return s

def to_csv_series(series: pd.Series, path: str, date_col: str = "date", value_col: str = "value"):
    out = series.rename(value_col).to_frame()
    out[date_col] = out.index
    out[[date_col, value_col]].to_csv(path, index=False)
