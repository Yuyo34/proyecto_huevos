# Upgrade de Precisión – Guía rápida

## 1) Instala dependencias mínimas
```
pip install -r upgrade_precision/requirements_min.txt
```

## 2) Prepara tus CSVs (formato `[date,value]` mensual `MS`)
- `precio_huevo_mensual.csv` (target)
- `usdclp.csv`, `ipc.csv`, `diesel_enap.csv`, `corn.csv`, `soy.csv` (opcionalmente los que tengas)

## 3) Corre el pipeline (ejemplo)
```
python -m upgrade_precision.pipeline.pipeline_monthly_exog           --target data/precio_huevo_mensual.csv           --usdclp data/usdclp.csv --ipc data/ipc.csv --diesel data/diesel_enap.csv           --corn data/corn.csv --soy data/soy.csv           --lags 1 2 3 --h 2 --out out/forecast_next2m.csv --multiplicative
```

## 4) Consejos de validación
- Ajusta `initial_window` en `rolling_backtest` (por defecto máx(24, 2*estacionalidad)).
- Revisa `bt["metrics"]` para ver si exógenas aportan (compara contra versión sin exógenas).
- Si la estacionalidad cambió tras shocks, re-estima con ventana móvil (modifica `stl_decompose`).

## 5) Ensamble con tus modelos existentes
Si tienes predicciones `ewma`, `drift`, `seasonal_naive` y `sarimax` para el mismo período de prueba, usa `eval/ensemble_weights.py` para aprender pesos y combinar.
```python
from upgrade_precision.eval.ensemble_weights import fit_weights, combine
weights = fit_weights(y_true, {"ewma": yhat_ewma, "drift": yhat_drift, "snaive": yhat_snaive, "sarimax": yhat_sarimax})
yhat_combo = combine({"ewma": yhat_ewma, "drift": yhat_drift, "snaive": yhat_snaive, "sarimax": yhat_sarimax}, weights)
```

## 6) Próximos pasos
- Añadir reconciliación jerárquica (MinT) para tipos/tamaños.
- Probar `Elastic Net` con pocas features como baseline ML y comparar.
