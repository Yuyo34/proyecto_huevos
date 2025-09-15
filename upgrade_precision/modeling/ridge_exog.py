import numpy as np
import pandas as pd

class RidgeExog:
    """
    Modelo ARX con regularización Ridge.
    """
    def __init__(self, alpha: float = 1.0, ar_lags=(1,12)):
        self.alpha = float(alpha)
        self.ar_lags = list(ar_lags)
        self.coef_ = None
        self.bias_ = None
        self.feat_names_ = None
        self.mu_ = None
        self.sd_ = None
        self.max_lag_ = max(self.ar_lags) if self.ar_lags else 0
        self.y_hist_ = None
        self.index_ = None

    def _make_matrix(self, y: pd.Series, X: pd.DataFrame | None):
        idx = y.index if X is None else y.index.intersection(X.index)
        df = pd.DataFrame(index=idx)
        for L in self.ar_lags:
            df[f"y_lag{L}"] = y.shift(L)
        if X is not None:
            Xc = X.reindex(idx)
            for c in Xc.columns:
                df[c] = Xc[c]
        df = df.dropna()
        y_vec = y.reindex(df.index).astype(float)
        mu = df.mean(axis=0)
        sd = df.std(axis=0, ddof=0).replace(0, 1.0)
        Xn = (df - mu) / sd
        return Xn, y_vec, mu, sd

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None):
        Xn, y_vec, mu, sd = self._make_matrix(y, X)
        ones = np.ones((len(Xn), 1))
        M = np.column_stack([ones, Xn.to_numpy(dtype=float)])
        p = M.shape[1]
        I = np.eye(p); I[0,0] = 0.0
        A = M.T @ M + self.alpha * I
        b = M.T @ y_vec.to_numpy(dtype=float)
        w = np.linalg.pinv(A) @ b
        self.bias_ = float(w[0])
        self.coef_ = w[1:]
        self.feat_names_ = list(Xn.columns)
        self.mu_ = mu
        self.sd_ = sd
        self.max_lag_ = max(self.ar_lags) if self.ar_lags else 0
        self.y_hist_ = y.copy()
        self.index_ = y.index
        return self

    def _predict_one(self, xf: pd.Series, y_hist: pd.Series):
        feats = {}
        for L in self.ar_lags:
            feats[f"y_lag{L}"] = float(y_hist.iloc[-L])
        if xf is not None:
            for c in xf.index:
                feats[c] = float(xf[c])
        f = pd.Series(feats, index=self.feat_names_).fillna(0.0)
        f_std = (f - self.mu_) / self.sd_
        return float(self.bias_ + np.dot(self.coef_, f_std.to_numpy(dtype=float)))

    def forecast(self, steps: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        last_ts = self.y_hist_.index[-1]
        future_idx = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=steps, freq="MS")

        if X_future is not None:
            Xf = X_future.copy()
            for c in self.feat_names_:
                if c not in Xf.columns:
                    Xf[c] = 0.0
            Xf = Xf[self.feat_names_]
        else:
            Xf = pd.DataFrame(0.0, index=future_idx, columns=self.feat_names_)

        y_hist = self.y_hist_.copy().astype(float)
        preds = []
        for t in future_idx:
            xf = Xf.loc[t] if Xf is not None else None
            if len(y_hist) < self.max_lag_:
                # antes: y_hist = y_hist.append(...)
                y_hist = pd.concat([y_hist, pd.Series([y_hist.iloc[-1]], index=[t])])
                preds.append(float(y_hist.iloc[-1]))
                continue
            yhat = self._predict_one(xf, y_hist)
            preds.append(yhat)
            # antes: y_hist = y_hist.append(...)
            y_hist = pd.concat([y_hist, pd.Series([yhat], index=[t])])
        return pd.Series(preds, index=future_idx, name="forecast")
