#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_gate.py (CSV amigable Excel opcional)
Reporte de calidad por estrato (último mes): n, IQR, n objetivo ±m, monotonicidad, bias histórico del imputado.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SIZE_ORDER = {"S":0,"M":1,"L":2,"XL":3}

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
    ap.add_argument("--raw", default="raw_prices_clean.csv")
    ap.add_argument("--mi", default="monthly_index_with_proxies.csv")
    ap.add_argument("--m_target", type=float, default=200.0, help="Semiancho del IC95% de la mediana (CLP).")
    ap.add_argument("--winsor", type=float, default=0.025)
    ap.add_argument("--excel-locale", action="store_true", default=False)
    ap.add_argument("--round-dp", type=int, default=0)
    return ap.parse_args()

def n_target_for_ci(iqr, m):
    if not np.isfinite(iqr) or iqr<=0 or m<=0: return np.nan
    return int(np.ceil(((1.82*iqr)/m)**2))

def check_monotonic(df_color):
    d = df_color.sort_values("egg_size", key=lambda s: s.map(SIZE_ORDER).fillna(99))
    ok = True
    prev = None
    for _, r in d.iterrows():
        if prev is not None and r["p50"] > prev + 1e-9:
            ok = False
        prev = r["p50"]
    return ok

def main():
    args = parse_args()
    base = Path(".")
    raw = pd.read_csv(base/args.raw, parse_dates=["date"])
    mi  = pd.read_csv(base/args.mi, dtype={"month":"string"})

    raw["month"] = raw["date"].dt.strftime("%Y-%m")
    g = (raw.groupby(["month","zone","egg_color","egg_size"])["price_clp"]
            .agg(n="count", p50="median",
                 p10=lambda s: s.quantile(0.10),
                 p90=lambda s: s.quantile(0.90))
            .reset_index())
    g["iqr"] = g["p90"] - g["p10"]

    last_m = g["month"].max()
    last = g[g["month"]==last_m].copy()
    last["n_target_m"] = last["iqr"].apply(lambda i: n_target_for_ci(i, args.m_target))

    mi["coverage_ok"] = mi.get("coverage_ok", False)
    if mi["coverage_ok"].dtype != bool:
        mi["coverage_ok"] = mi["coverage_ok"].astype(str).str.lower().isin(["true","1","yes","si","sí"])
    for c in ["p50","wholesale_imputed"]:
        if c not in mi: mi[c] = np.nan
        mi[c] = pd.to_numeric(mi[c], errors="coerce")
    hist = mi[mi["coverage_ok"] & mi["p50"].notna() & mi["wholesale_imputed"].notna()].copy()
    hist["bias_pct"] = 100.0*(hist["p50"]-hist["wholesale_imputed"])/hist["wholesale_imputed"]
    bias = (hist.groupby(["zone","egg_color","egg_size"])["bias_pct"]
                .agg(bias_median_pct="median", bias_mean_pct="mean", bias_n="count")
                .reset_index())

    # Monotonicidad por color (último mes)
    mono = []
    for (zone, col), sub in last.groupby(["zone","egg_color"]):
        ok = check_monotonic(sub)
        mono.append(dict(zone=zone, egg_color=col, monotonic_ok=ok))
    mono = pd.DataFrame(mono)

    report = last.merge(bias, on=["zone","egg_color","egg_size"], how="left") \
                 .merge(mono, on=["zone","egg_color"], how="left") \
                 .sort_values(["zone","egg_color","egg_size"])

    write_outputs(report, Path("quality_report.csv"), excel_locale=args.excel_locale, round_dp=args.round_dp)

    summary = {
        "last_month": last_m,
        "estratos": int(report.shape[0]),
        "share_n_ge_7": float((report["n"]>=7).mean() if "n" in report else 0),
        "estratos_monot_ok": int(report["monotonic_ok"].sum() if "monotonic_ok" in report else 0),
        "estratos_bias_hi_gt_10pct": int((report["bias_median_pct"].abs()>10).sum() if "bias_median_pct" in report else 0),
    }
    Path("quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK -> quality_report.csv | resumen -> quality_summary.json")

if __name__ == "__main__":
    main()
