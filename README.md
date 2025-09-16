# proyecto_huevos

Pronóstico **mensual** del precio del huevo en **CLP reales (Chile)** con señales exógenas.  
Base ganadora actual: **% importados (YoY) FOR_TARGET, lags=1, bt_init=12** → **MASE ≈ 0.9512** (modelo elegido: SARIMAX, h=2).

## Requisitos
- Python 3.10+ (sugerido 3.11)
- Paquetes: numpy, pandas, scikit-learn, statsmodels, pmdarima, matplotlib, pdfplumber, python-dateutil, unidecode

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

@'
...contenido...
