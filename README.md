# Proyecto de pronóstico mensual del precio mayorista del huevo

Este repositorio contiene el flujo completo para generar pronósticos mensuales del precio mayorista del huevo en Chile (CLP por docena). El pipeline toma nuevas cotizaciones recopiladas por el equipo, las normaliza, construye índices mayoristas por zona/color/tamaño, calibra el traspaso retail→mayorista con series de ODEPA y finalmente proyecta los próximos meses con bandas de incertidumbre y un reporte de calidad.【F:pipeline_monthly.py†L4-L101】【F:forecast_next_2m.py†L3-L283】

## Tabla de contenidos
1. [Requisitos previos](#requisitos-previos)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Archivos de configuración](#archivos-de-configuración)
4. [Actualización mensual paso a paso](#actualización-mensual-paso-a-paso)
5. [Salidas principales](#salidas-principales)
6. [Comandos útiles individuales](#comandos-útiles-individuales)
7. [Resolución de problemas y buenas prácticas](#resolución-de-problemas-y-buenas-prácticas)

## Requisitos previos
- Python 3.10 o superior.
- Entorno virtual recomendado (`python -m venv .venv` y activar antes de instalar dependencias).
- Librerías mínimas: `pandas`, `numpy`, `statsmodels` y `scikit-learn`. Instálalas con `pip install -r upgrade_precision/requirements_min.txt`.【F:upgrade_precision/requirements_min.txt†L1-L4】
- `pyyaml` para leer la configuración (`pip install pyyaml`).

## Estructura del repositorio
Los archivos clave para la operación mensual son:

| Ruta | Descripción |
| --- | --- |
| `config.yml` | Define rutas de entrada/salida y parámetros numéricos usados por todo el pipeline.【F:config.yml†L1-L20】 |
| `inputs/anchors/` | Carpeta donde se depositan los CSV de cotizaciones nuevas a ingerir.【F:ingest_month.py†L73-L88】 |
| `anchors_wholesale.csv` | Maestro que acumula todas las observaciones históricas luego de la ingesta.【F:ingest_month.py†L95-L114】 |
| `raw_prices_clean.csv` | Precios mayoristas limpios (puede generarse con `anchors_to_raw.py`).【F:pipeline_monthly.py†L45-L49】 |
| `odepa_retail.csv` | Serie retail mensual por estrato descargada desde ODEPA.【F:build_wholesale_index.py†L203-L220】 |
| `pipeline_monthly.py` | Orquestador que ejecuta cada etapa de la actualización mensual.【F:pipeline_monthly.py†L4-L101】 |
| `logs/` | Directorio sugerido para capturar salidas si automatizas la ejecución (no se genera automáticamente). |

Además, la carpeta `upgrade_precision/` contiene utilidades genéricas de modelamiento usadas por algunos scripts complementarios.

## Archivos de configuración
El archivo `config.yml` agrupa tres bloques:

- **paths**: rutas relativas a los principales insumos y productos (archivos CSV de entrada/salida).
- **params**: hiperparámetros numéricos compartidos, como `kmin` (mínimo de observaciones para considerar cobertura aceptable), `default_ratio` (ratio retail→mayorista por defecto) o el horizonte `forecast_h` para las proyecciones.【F:config.yml†L1-L20】
- **anchors_schema**: columnas mínimas exigidas en los CSV crudos entregados por el equipo de campo.【F:config.yml†L21-L22】

Adapta rutas o parámetros a tus necesidades antes de correr el pipeline.

## Actualización mensual paso a paso
Ejecuta todo el flujo con un solo comando:

```bash
python pipeline_monthly.py --config config.yml
```

El orquestador realiza las siguientes etapas en orden y detiene la ejecución si alguna falla. Cada paso imprime el comando ejecutado y reutiliza los parámetros definidos en `config.yml`.【F:pipeline_monthly.py†L36-L101】

1. **Ingesta de anchors** – Combina todos los CSV en `inputs/anchors/`, normaliza columnas (incluye conversión automática de coma decimal) y actualiza `anchors_wholesale.csv`. Los archivos temporales de Excel se ignoran automáticamente.【F:ingest_month.py†L25-L114】
2. **Conversión a precios crudos** – Si existe `anchors_to_raw.py`, transforma los anchors a `raw_prices.csv`/`raw_prices_clean.csv` aplicando reglas propias del equipo.【F:pipeline_monthly.py†L45-L49】
3. **Backfill opcional** – `backfill_from_proxies.py` (si está presente) permite imputar huecos antes de construir el índice.【F:pipeline_monthly.py†L51-L55】
4. **Índice mayorista** – `build_wholesale_index.py` limpia unidades, aplica winsorization por estrato, realiza pooling jerárquico, fuerza monotonicidad por tamaño y genera `monthly_index.csv` junto a `monthly_index_with_proxies.csv` (incluye retail/imputado y razón utilizada).【F:build_wholesale_index.py†L4-L240】
5. **Factores estacionales** – `seasonal_factors.py` resume estacionalidad por zona a partir de la serie retail (genera CSV estándar y variante amigable para Excel).【F:pipeline_monthly.py†L68-L72】
6. **Calibración de ratios** – `calibrate_ratios.py` aprende ratios retail→mayorista con backtesting EWMA, entrega `ratios_calibrated.csv` y un resumen JSON con métricas de ajuste.【F:calibrate_ratios.py†L4-L193】
7. **Pronóstico** – `forecast_next_2m.py` decide automáticamente entre extrapolación directa y método retail-ratio, usa los ratios calibrados y los factores estacionales para producir `forecast_1_2m.csv` con p10/p50/p90 por estrato.【F:forecast_next_2m.py†L3-L283】
8. **Quality Gate** – `quality_gate.py` arma `quality_report.csv` (y versión Excel si corresponde) con conteos, IQR, metas de tamaño de muestra, chequeo de monotonicidad y sesgo histórico del imputado. También escribe `quality_summary.json`.【F:quality_gate.py†L4-L109】

## Salidas principales
Tras ejecutar el pipeline encontrarás, como mínimo, los siguientes archivos en la raíz del proyecto:

| Archivo | Contenido clave |
| --- | --- |
| `monthly_index.csv` | Medianas, cuantiles y banderas de cobertura por mes/zona/color/tamaño luego de limpiar y poolizar los precios.【F:build_wholesale_index.py†L178-L201】 |
| `monthly_index_with_proxies.csv` | Igual que el anterior pero agrega retail, ratio aplicado, valor imputado y delta porcentual frente a la mediana observada.【F:build_wholesale_index.py†L203-L240】 |
| `seasonal_factors.csv` | Factores estacionales por zona (y otros niveles si los agregas) usados para desestacionalizar/reestacionalizar pronósticos.【F:pipeline_monthly.py†L68-L72】【F:forecast_next_2m.py†L203-L269】 |
| `ratios_calibrated.csv` | Tabla por estrato con ratio global, ratio EWMA, mezcla final y métricas de backtesting; acompaña `ratios_summary.json` con estadísticas de control.【F:calibrate_ratios.py†L130-L193】 |
| `forecast_1_2m.csv` | Pronósticos p50 y bandas empíricas (p10/p90) para los próximos meses, especificando el método elegido en cada caso.【F:forecast_next_2m.py†L244-L283】 |
| `quality_report.csv` | Reporte del último mes con conteos, IQR, meta de n para IC95%, monotonicidad y sesgo vs. imputado; incluye resumen JSON.【F:quality_gate.py†L64-L109】 |

Si `excel_locale` está en `true`, cada CSV numérico genera además una copia `_excel.csv` con separador `;` y coma decimal pensada para configuraciones regionales de Excel.【F:pipeline_monthly.py†L39-L40】【F:build_wholesale_index.py†L35-L38】

## Comandos útiles individuales
Aunque el pipeline cubre todo el flujo, puedes ejecutar scripts sueltos cuando necesites depurar alguna etapa:

- Ingesta sin correr el resto:
  ```bash
  python ingest_month.py --config config.yml
  ```
  Ideal para revisar que los nuevos anchors se integran correctamente antes de continuar.【F:ingest_month.py†L20-L114】

- Reconstruir solo el índice mayorista:
  ```bash
  python build_wholesale_index.py --raw raw_prices_clean.csv --odepa odepa_retail.csv --ratios ratios_calibrated.csv --excel-locale --round-dp 0
  ```
  Útil cuando ajustas parámetros de winsorization, pooling o ratio por defecto.【F:build_wholesale_index.py†L4-L240】

- Recalibrar ratios luego de corregir la serie retail:
  ```bash
  python calibrate_ratios.py --mi monthly_index_with_proxies.csv --odepa odepa_retail.csv --out ratios_calibrated.csv
  ```
  Genera también `ratios_summary.json` con conteos de estratos backtesteados.【F:calibrate_ratios.py†L32-L193】

- Reemitir pronósticos usando un horizonte distinto:
  ```bash
  python forecast_next_2m.py --mi monthly_index_with_proxies.csv --odepa odepa_retail.csv --ratios ratios_calibrated.csv --seasonal seasonal_factors.csv --h 2 --alpha 0.35
  ```
  Ajusta `--h` y `--alpha` según tus necesidades sin volver a correr etapas previas.【F:forecast_next_2m.py†L33-L283】

## Resolución de problemas y buenas prácticas
- **Validar entradas**: antes de correr el pipeline confirma que `raw_prices_clean.csv` y `odepa_retail.csv` tengan las columnas requeridas (`date`, `zone`, `egg_color`, `egg_size`, `unit`, `price_clp`, etc.). Los scripts detendrán la ejecución si faltan datos clave.【F:build_wholesale_index.py†L161-L214】
- **Cobertura mínima**: los parámetros `kmin` y `kpool` controlan la robustez de las medianas. Si ves muchos `coverage_ok = False`, considera recolectar más observaciones o relajar `kmin` en `config.yml`.【F:build_wholesale_index.py†L178-L199】【F:config.yml†L12-L14】
- **Ratios faltantes**: cuando no hay ratio calibrado para un estrato se usa el valor por defecto (`default_ratio`). Ajusta este parámetro si detectas sesgos sistemáticos.【F:build_wholesale_index.py†L222-L236】【F:config.yml†L15-L16】
- **Revisión de calidad**: revisa siempre `quality_report.csv` y `quality_summary.json` tras cada corrida para decidir si el entregable puede publicarse o si se requiere depuración adicional de datos.【F:quality_gate.py†L72-L109】
- **Registro de ejecuciones**: si automatizas el pipeline (por ejemplo con `cron` o GitHub Actions) redirige la salida estándar a `logs/` para auditoría futura.

Con esta guía deberías poder entender el flujo de datos, modificar parámetros y ejecutar la actualización mensual de punta a punta.
