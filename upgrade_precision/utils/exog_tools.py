from __future__ import annotations
import pandas as pd
import numpy as np

def build_exog_matrix(target: pd.Series,
                      exog_dict: dict[str, pd.Series],
                      lags: list[int] = [1, 2, 3],
                      log_transform: list[str] | None = None) -> pd.DataFrame:
    log_transform = set(log_transform or [])
    X = pd.DataFrame(index=target.index)
    for name, ser in exog_dict.items():
        s = ser.copy().astype(float).sort_index()
        s = s.reindex(target.index).interpolate(limit_direction="both")
        if name in log_transform:
            s = np.log(s.clip(lower=1e-9))
        for lag in lags:
            X[f"{name}_lag{lag}"] = s.shift(lag)
    X = X.loc[target.index]
    return X
