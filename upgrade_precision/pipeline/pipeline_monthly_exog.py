from __future__ import annotations
      import argparse
      import pandas as pd
      import numpy as np
      from pathlib import Path

      from ..utils.io_tools import read_series_csv, to_csv_series
      from ..utils.exog_tools import build_exog_matrix
      from ..utils.stl_tools import stl_decompose, reseasonalize, deseasonalize
      from ..utils.metrics import mape, smape, mase
      from ..modeling.sarimax_exog import SarimaxExog
      from ..eval.backtesting import rolling_backtest
      from ..eval.ensemble_weights import fit_weights, combine

      """
      Ejemplo de uso:
      python -m upgrade_precision.pipeline.pipeline_monthly_exog \
--target data/precio_huevo_mensual.csv \
--usdclp data/usdclp.csv --ipc data/ipc.csv --diesel data/diesel_enap.csv \
--corn data/corn.csv --soy data/soy.csv \
--lags 1 2 3 --h 2 --out forecasts.csv

      Formato esperado de cada CSV: columnas [date,value], freq mensual (MS).
      """

      BASE_REGRESSORS = ["usdclp", "ipc", "diesel", "corn", "soy"]

      def main():
          ap = argparse.ArgumentParser()
          ap.add_argument("--target", required=True)
          for r in BASE_REGRESSORS:
              ap.add_argument(f"--{r}", required=False)
          ap.add_argument("--lags", nargs="+", type=int, default=[1,2,3])
          ap.add_argument("--seasonality", type=int, default=12)
          ap.add_argument("--h", type=int, default=2)
          ap.add_argument("--out", required=True)
          ap.add_argument("--multiplicative", action="store_true")
          args = ap.parse_args()

          y = read_series_csv(args.target)

          exog = {}
          for r in BASE_REGRESSORS:
              path = getattr(args, r)
              if path:
                  exog[r] = read_series_csv(path)
          X = build_exog_matrix(y, exog, lags=args.lags, log_transform=["usdclp", "diesel", "corn", "soy"]) if exog else None

          trend, seasonal, resid = stl_decompose(y, period=args.seasonality, robust=True)
          y_deseas = deseasonalize(y, seasonal, multiplicative=args.multiplicative).dropna()

          def builder():
              return SarimaxExog(seasonal_period=args.seasonality,
                                 pdq_grid=[(0,1,1),(1,1,0),(1,1,1)],
                                 PDQ_grid=[(0,1,1),(1,1,0)],
                                 trend=None)

          bt = rolling_backtest(y_deseas, X.loc[y_deseas.index] if X is not None else None,
                                builder, horizon=1, initial_window=max(24, args.seasonality*2))
          print("Métricas in-sample (deseas):", bt["metrics"])

          model = builder()
          model.fit(y_deseas, X.loc[y_deseas.index] if X is not None else None)

          last_idx = y.index[-1]
          future_idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=args.h, freq="MS")

          if X is not None:
              X_ext = X.copy()
              last_row = X_ext.iloc[[-1]].to_numpy()
              fut = np.repeat(last_row, args.h, axis=0)
              X_future = pd.DataFrame(fut, index=future_idx, columns=X.columns)
          else:
              X_future = None

          deseas_forecast = model.forecast(steps=args.h, X_future=X_future)
          seasonal_future = seasonal.reindex(future_idx)
          y_forecast = reseasonalize(deseas_forecast, seasonal_future, multiplicative=args.multiplicative)

          out = pd.Series(y_forecast, index=future_idx, name="forecast")
          out_df = out.to_frame()
          out_df["date"] = out_df.index
          out_df = out_df[["date","forecast"]]
          out_df.to_csv(args.out, index=False)
          print(f"Pronóstico guardado en {args.out}")

      if __name__ == "__main__":
          main()
