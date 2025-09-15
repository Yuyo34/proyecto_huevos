## Cómo correr el pronóstico

Requisitos: Python 3.10+ y dependencias de \upgrade_precision/requirements_min.txt\.

1. Preparar datos exógenos (si se usan): \data/usdclp_dlog.csv\, \data/ipc_yoy.csv\ (formato \date,value\ mensual MS).
2. Ejecutar:
   \\\ash
   py -m upgrade_precision.pipeline.pipeline_monthly_exog ^
     --target data\precio_huevo_mensual.csv ^
     --usdclp data\usdclp_dlog.csv --ipc data\ipc_yoy.csv ^
     --lags 1 --bt_init 12 --seasonality 12 --h 2 ^
     --no_boost ^
     --out out\forecast_next2m.csv
   \\\
3. También puedes usar \un_forecast.bat\.

Notas:
- El modelo baseline hoy es el ganador (MASE ~0.825).
- El booster de residuo está disponible pero desactivado por defecto (\--no_boost\).
