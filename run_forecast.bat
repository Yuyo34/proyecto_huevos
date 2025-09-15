@echo off
setlocal
cd /d "%~dp0"
py -m upgrade_precision.pipeline.pipeline_monthly_exog ^
  --target "data\precio_huevo_mensual.csv" ^
  --usdclp "data\usdclp_dlog.csv" --ipc "data\ipc_yoy.csv" ^
  --lags 1 --bt_init 12 --seasonality 12 --h 2 ^
  --no_boost ^
  --out "out\forecast_next2m.csv"
echo Hecho. Salida: out\forecast_next2m.csv
endlocal
