#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seasonal_factors.py (CSV amigable Excel opcional)
Calcula factores estacionales multiplicativos (mes del año) desde ODEPA.
Nivel por zona y, si hay historia suficiente (>=24 meses), por zona×color y zona×color×tamaño.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -------- Excel helpers --------
def _round_numeric(df, dp=0):
    if dp is None: return df
    out = df.copy()
    for c in out.select_dtypes(include=[float, int]).columns:
        out[c] = out[c].round(int(dp))
    return out

def _to_excel_csv(df, path, dp=0):
    df2 = _round_numeric(df, dp=dp).copy()
    for c in df2.select_dtypes(include=[float, int]).columns:
        df2[c] = df2[c].map(lambda x: ('' if pd.isna(x) else str(x).replace('.', ',')))
    df2.to_csv(path, sep=';', index=False, encoding='utf-8-sig')

def write_outputs(df, path: Path, excel_locale=False, round_dp=0):
    df.to_csv(path, index=False, encoding='utf-8')
    if excel_locale:
        _to_excel_csv(df, Path(str(path).replace('.csv','_excel.csv')), dp=round_dp)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odepa", default="odepa_retail.csv")
    ap.add_argument("--out", default="seasonal_factors.csv")
    ap.add_argument("--min_months_stratum", type=int, default=24)
    ap.add_argument("--excel-locale", action="store_true", default=False)
    ap.add_argument("--round-dp", type=int, default=3)
    return ap.parse_args()

def factors_for_group(df):
    df = df.dropna(subset=["retail_price_clp_per_dozen","month"]).copy()
    if df.empty: return None
    df["moy"] = df["month"].str.slice(5,7).astype(int)
    overall = float(df["retail_price_clp_per_dozen"].median())
    fac = df.groupby("moy")["retail_price_clp_per_dozen"].median()/overall if overall>0 else np.nan
    series = {f"m{m:02d}": float(fac.get(m, 1.0)) if pd.notna(fac.get(m, np.nan)) else 1.0 for m in range(1,13)}
    series["base_median"] = overall
    return series

def main():
    args = parse_args()
    base = Path(".")
    od = pd.read_csv(base/args.odepa, parse_dates=["date"])
    od["month"] = od["date"].dt.strftime("%Y-%m")

    rows = []
    # Por ZONA
    for z, g in od.groupby("zone"):
        fac = factors_for_group(g)
        if fac: rows.append(dict(level="zone", zone=z, **fac))

    # Por ZONA×COLOR
    for (z,c), g in od.groupby(["zone","egg_color"]):
        if g["month"].nunique() >= args.min_months_stratum:
            fac = factors_for_group(g)
            if fac: rows.append(dict(level="zone_color", zone=z, egg_color=c, **fac))

    # Por ZONA×COLOR×TAMAÑO
    for (z,c,s), g in od.groupby(["zone","egg_color","egg_size"]):
        if g["month"].nunique() >= args.min_months_stratum:
            fac = factors_for_group(g)
            if fac: rows.append(dict(level="zone_color_size", zone=z, egg_color=c, egg_size=s, **fac))

    out = pd.DataFrame(rows).sort_values(["level","zone","egg_color","egg_size"], na_position="last")
    write_outputs(out, base/args.out, excel_locale=args.excel_locale, round_dp=args.round_dp)
    print("OK ->", args.out)

if __name__ == "__main__":
    main()
