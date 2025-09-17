# Proyecto: Pronóstico mensual del precio mayorista del huevo (docena, CLP)

## Objetivo
Pronosticar el precio mayorista mensual por docena en CLP. Se generan horizontes h=1 y h=2 y un entregable para gerencia.

## Datos
- TARGET: `data/precio_huevo_mensual_real.csv` (tiene huecos: 21 de 61 meses entre 2020-09 y 2025-09).
- Exógenas opcionales: `usdclp_dlog.csv`, `imacec_yoy.csv`, `pct_imp_yoy_FOR_TARGET.csv`.
- IPC opcional: `ipc_index.csv` (para convertir el entregable a nominal).

## Cómo correr (PowerShell)
```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m upgrade_precision.pipeline.pipeline_monthly_exog `
  --target data\precio_huevo_mensual_real.csv `
  --usdclp data\usdclp_dlog.csv --ipc data\imacec_yoy.csv `
  --soy data\pct_imp_yoy_FOR_TARGET.csv `
  --lags 1 --bt_init 12 --seasonality 12 --no_boost --h 1 `
  --out out\fcst_base_h1.csv
& $PY -m upgrade_precision.pipeline.pipeline_monthly_exog ... --h 2 --out out\fcst_base_h2.csv
& $PY -u scripts\cv_rolling_eval_fast.py
& $PY -u scripts\rocv_score.py
# opcional (nominal):
& $PY -u scripts\entregable_nominal_v2.py
Entregable

out/entregable_nominal.csv y out/entregable_nominal.png (si hay IPC).

Unidades: mismas del TARGET (docena, CLP).

Pendientes

Completar meses faltantes con datos reales (ODEPA/Lo Valledor/INE/BCCh).

Parser de ODEPA y fetcher de IPC. 

