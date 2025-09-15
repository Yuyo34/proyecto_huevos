import numpy as np, pandas as pd

class ResidualRidge:
    """
    Ajusta y_resid = y - y_hat_base ~ Ridge(features).
    Features deben venir ya alineadas (mismo índice que y_resid).
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.bias_ = 0.0
        self.cols_ = None
        self.mu_ = None
        self.sd_ = None

    def fit(self, y_resid: pd.Series, X: pd.DataFrame):
        df = X.reindex(y_resid.index).dropna()
        yv = y_resid.reindex(df.index).astype(float)
        mu = df.mean(); sd = df.std(ddof=0).replace(0, 1.0)
        Z = (df - mu) / sd
        import numpy as np
        ones = np.ones((len(Z),1))
        M = np.column_stack([ones, Z.to_numpy(float)])
        import numpy as np
        I = np.eye(M.shape[1]); I[0,0]=0.0
        w = np.linalg.pinv(M.T @ M + self.alpha*I) @ (M.T @ yv.to_numpy(float))
        self.bias_ = float(w[0]); self.coef_ = w[1:]
        self.cols_ = list(df.columns); self.mu_ = mu; self.sd_ = sd
        # Diagnóstico: R^2 in-sample sobre el residuo
        yhat = M @ w
        sse = float(((yv.to_numpy(float) - yhat)**2).sum())
        sst = float(((yv.to_numpy(float) - yv.mean())**2).sum()) or 1.0
        self.r2_ = 1.0 - sse/sst
        return self

    def forecast(self, X_future: pd.DataFrame) -> pd.Series:
        Xf = X_future.copy()
        for c in self.cols_:
            if c not in Xf.columns:
                Xf[c] = 0.0
        Xf = Xf[self.cols_]
        Z = (Xf - self.mu_) / self.sd_
        y = self.bias_ + Z.to_numpy(float) @ self.coef_
        return pd.Series(y, index=Xf.index, name="resid_boost")

