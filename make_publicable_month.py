#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera 'precio_publicable_<YYYY-MM>.csv' desde monthly_index_with_proxies.csv

Reglas:
- precio_publicable = p50 si coverage_ok==True; en caso contrario usa wholesale_imputed.
- Incluye rango (p10, p90), n, delta_vs_imputed_pct para control.
- Filtra por mes (--month YYYY-MM). Si no se indica, usa el último mes disponible.
- Filtra por zonas con --zones "Región Metropolitana,Región de Valparaíso" (opcional).
"""
import argparse, sys
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=str, default=None, help="Mes YYYY-MM (si no, usa el último).")
    ap.add_argument("--zones", type=str, default=None, help='Zonas separadas por coma (exactamente como en el CSV).')
    ap.add_argument("--infile", type=str, default="monthly_index_with_proxies.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    path = BASE / args.infile
    if not path.exists():
        print(f"Falta {path}. Corre antes build_wholesale_index.py.")
        sys.exit(1)

    df = pd.read_csv(path, dtype={"month":"string"})
    if "month" not in df.columns:
        print("El archivo no tiene columna 'month'.")
        sys.exit(2)

    # Determinar mes
    month = args.month or df["month"].dropna().max()
    outname = f"precio_publicable_{month}.csv"

    out = df[df["month"] == month].copy()
    if args.zones:
        zones = [z.strip() for z in args.zones.split(",") if z.strip()]
        out = out[out["zone"].isin(zones)]

    # Columnas esperadas
    needed = ["zone","egg_color","egg_size","p50","p10","p90","n","coverage_ok","wholesale_imputed","delta_vs_imputed_pct"]
    for c in needed:
        if c not in out.columns:
            out[c] = None

    # Precio publicable
    def pick_price(r):
        if str(r.get("coverage_ok", "")).lower() in ("true","1","yes","si","sí"):
            return r.get("p50")
        return r.get("wholesale_imputed")

    out["precio_publicable"] = out.apply(pick_price, axis=1)

    # Orden y redondeo
    cols = ["month","zone","egg_color","egg_size",
            "precio_publicable","p50","p10","p90","n","coverage_ok","wholesale_imputed","delta_vs_imputed_pct"]
    for c in ["precio_publicable","p50","p10","p90","wholesale_imputed","delta_vs_imputed_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[cols].sort_values(["zone","egg_color","egg_size"])

    # Redondeos (ajusta si prefieres)
    out["precio_publicable"] = out["precio_publicable"].round(0)
    for c in ["p50","p10","p90","wholesale_imputed"]:
        out[c] = out[c].round(0)
    if "delta_vs_imputed_pct" in out:
        out["delta_vs_imputed_pct"] = out["delta_vs_imputed_pct"].round(1)

    out.to_csv(BASE / outname, index=False, encoding="utf-8")
    print(f"OK -> {outname}")
    
    out["CLP_por_huevo"] = (out["precio_publicable"] / 12.0).round(1)
    
if __name__ == "__main__":
    main()

