#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_monthly.py
Orquesta la actualización mensual:
  1. Ingesta de nuevos anchors (inputs/anchors/*.csv) -> anchors_wholesale.csv
  2. anchors_to_raw.py -> raw_prices.csv
  3. backfill_from_proxies.py (si aplica) -> backfilled_wholesale.csv
  4. build_wholesale_index.py -> monthly_index(_excel).csv + monthly_index_with_proxies(_excel).csv
  5. seasonal_factors.py -> seasonal_factors(_excel).csv
  6. calibrate_ratios.py -> ratios_calibrated(_excel).csv
  7. forecast_next_2m.py -> forecast_1_2m(_excel).csv
  8. quality_gate.py -> quality_report(_excel).csv

Uso:
  python pipeline_monthly.py --config config.yml
"""

import argparse, subprocess, sys
from pathlib import Path
import yaml

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    return ap.parse_args()

def run(cmd):
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    excel_flag = ["--excel-locale"] if cfg["params"].get("excel_locale", False) else []
    round_flag = ["--round-dp", str(cfg["params"].get("round_dp", 0))]

    # 1) Ingesta
    run(["python", "ingest_month.py", "--config", args.config])

    # 2) anchors_to_raw
    if Path("anchors_to_raw.py").exists():
        run(["python", "anchors_to_raw.py"])
    else:
        print("[WARN] anchors_to_raw.py no encontrado, salto este paso.")

    # 3) Backfill
    if Path("backfill_from_proxies.py").exists():
        run(["python", "backfill_from_proxies.py"])
    else:
        print("[WARN] backfill_from_proxies.py no encontrado, salto este paso.")

    # 4) Índice
    run(["python", "build_wholesale_index.py",
         "--raw", cfg["paths"]["raw_prices_clean"],
         "--odepa", cfg["paths"]["odepa_retail"],
         "--ratios", cfg["paths"]["ratios_calibrated"],
         "--kmin", str(cfg["params"]["kmin"]),
         "--kpool", str(cfg["params"]["kpool"]),
         "--winsor", str(cfg["params"]["winsor"]),
         "--ratio", str(cfg["params"]["default_ratio"]),
         *excel_flag, *round_flag])

    # 5) Estacionalidad
    run(["python", "seasonal_factors.py",
         "--odepa", cfg["paths"]["odepa_retail"],
         "--out", cfg["paths"]["seasonal_factors"],
         *excel_flag, "--round-dp", "3"])

    # 6) Ratios calibrados
    run(["python", "calibrate_ratios.py",
         "--mi", cfg["paths"]["monthly_index_with_proxies"],
         "--odepa", cfg["paths"]["odepa_retail"],
         "--out", cfg["paths"]["ratios_calibrated"],
         *excel_flag, "--round-dp", "4"])

    # 7) Forecast
    run(["python", "forecast_next_2m.py",
         "--mi", cfg["paths"]["monthly_index_with_proxies"],
         "--odepa", cfg["paths"]["odepa_retail"],
         "--ratios", cfg["paths"]["ratios_calibrated"],
         "--seasonal", cfg["paths"]["seasonal_factors"],
         "--h", str(cfg["params"]["forecast_h"]),
         "--alpha", str(cfg["params"]["forecast_alpha"]),
         "--default-ratio", str(cfg["params"]["default_ratio"]),
         "--out", cfg["paths"]["forecast_out"],
         *excel_flag, *round_flag])

    # 8) Quality gate
    run(["python", "quality_gate.py",
         "--raw", cfg["paths"]["raw_prices_clean"],
         "--mi", cfg["paths"]["monthly_index_with_proxies"],
         "--m_target", str(cfg["params"]["m_target_ci95"]),
         *excel_flag, *round_flag])

    print("OK -> Pipeline mensual completo. Revisa los _excel.csv si usas Excel regional.")

if __name__ == "__main__":
    main()
