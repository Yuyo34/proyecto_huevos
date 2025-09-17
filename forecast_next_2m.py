#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_next_2m.py (con estacionalidad + CSV amigable Excel)
Elige automáticamente entre método directo y retail→ratio, con bandas empíricas.
Soporta desestacionalización por zona (si se entrega seasonal_factors.csv).
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
    ap.add_argument("--mi", default="monthly_index_with_proxies.csv")
    ap.add_argument("--odepa", default="odepa_retail.csv")
    ap.add_argument("--ratios", default="ratios_calibrated.csv")
    ap.add_argument("--seasonal", default="seasonal_factors.csv")
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--default-ratio", type=float, default=0.85)
    ap.add_argument("--out", default="forecast_1_2m.csv")
    ap.add_argument("--excel-locale", action="store_true", default=False)
    ap.add_argument("--round-dp", type=int, default=0)
    return ap.parse_args()

def add_months(ym: str, k: int) -> str:
    y, m = map(int, ym.split("-"))
    m2 = m + k
    y += (m2-1)//12
    m = ((m2-1)%12)+1
    return f"{y:04d}-{m:02d}"

def ewma_level(series, alpha=0.35):
    s = [x for x in series if pd.notna(x)]
    if not s: return np.nan
    level = float(s[0])
    for x in s[1:]:
        level = alpha*float(x) + (1-alpha)*level
    return level

def drift_forecast(series, h=1):
    s = [x for x in series if pd.notna(x)]
    n = len(s)
    if n == 0: return np.nan
    if n == 1: return float(s[-1])
    drift = (float(s[-1]) - float(s[0]))/(n-1)
    return float(s[-1] + h*drift)

def seasonal_naive(months, series, h=1):
    if len(series) == 0: return np.nan
    last_m = months[-1]
    target = add_months(last_m, h)
    lag12 = add_months(target, -12)
    m2v = {m: v for m, v in zip(months, series)}
    if lag12 in m2v and pd.notna(m2v[lag12]): return float(m2v[lag12])
    for v in reversed(series):
        if pd.notna(v): return float(v)
    return np.nan

def ensemble_point(months, series, h=1, alpha=0.35):
    a = ewma_level(series, alpha=alpha)
    b = drift_forecast(series, h=h)
    c = seasonal_naive(months, series, h=h)
    vals = [x for x in [a,b,c] if pd.notna(x)]
    if not vals: return np.nan
    return float(np.mean(vals))

def mdape(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    mask = (np.isfinite(y) & np.isfinite(yhat) & (y!=0))
    if not mask.any(): return np.nan
    return np.median(np.abs((yhat[mask]-y[mask]) / y[mask]))*100.0

def empirical_band(months, series, alpha=0.35):
    s = pd.Series(series, index=months).dropna()
    if len(s) < 6:
        m = np.nanmean(series) or 0.0
        return (-0.05*m, 0.05*m)
    errs = []
    mm = list(s.index)
    for i in range(3, len(s)):
        yhat = ensemble_point(mm[:i], s.iloc[:i].values, h=1, alpha=alpha)
        errs.append(float(s.iloc[i] - yhat))
    q10, q90 = np.percentile(errs, [10, 90]) if errs else (-0.05*s.median(), 0.05*s.median())
    return (q10, q90)

# Seasonal helpers
def read_seasonal(path: Path):
    if not path.exists(): return None
    return pd.read_csv(path)

def factor_for(sf, month, z):
    if sf is None: return 1.0
    moy = int(month.split("-")[1]); col = f"m{moy:02d}"
    m = sf[(sf.get("level")=="zone") & (sf.get("zone")==z)]
    if not m.empty and col in m.columns and pd.notna(m.iloc[0][col]):
        return float(m.iloc[0][col])
    return 1.0

def choose_method(months, y, retail, z, c, s, ratios, default_ratio, alpha=0.35):
    if len(y) < 8: return "direct"
    preds_d, trues = [], []
    for i in range(6, len(y)-1):
        yhat = ensemble_point(months[:i+1], y[:i+1], h=1, alpha=alpha)
        preds_d.append(yhat); trues.append(y[i+1])
    score_d = mdape(trues, preds_d)

    preds_r = []
    if retail is not None and len(retail)==len(y):
        ratio_value = default_ratio
        if ratios is not None:
            ratio_value, _ = pick_ratio_value(z, c, s, ratios, default_ratio)
            if not np.isfinite(ratio_value):
                ratio_value = default_ratio
        for i in range(6, len(y)-1):
            r_hat = ensemble_point(months[:i+1], retail[:i+1], h=1, alpha=alpha)
            preds_r.append(r_hat*ratio_value if pd.notna(r_hat) else np.nan)
        score_r = mdape(trues, preds_r) if preds_r else np.nan
    else:
        score_r = np.nan

    if np.isfinite(score_r) and score_r + 0.1 < score_d:
        return "retail_ratio"
    return "direct"

def load_ratios_maps(path: Path):
    if not path.exists(): return None
    r = pd.read_csv(path)
    if "ratio_blend" not in r.columns: return None
    m_exact = {(row["zone"], row["egg_color"], row["egg_size"]): row["ratio_blend"]
               for _, row in r.dropna(subset=["ratio_blend"]).iterrows()}
    m_zc = {}
    m_z  = {}
    for (z,c), g in r.groupby(["zone","egg_color"]):
        m_zc[(z,c)] = float(g["ratio_blend"].median())
    for z, g in r.groupby(["zone"]):
        m_z[z] = float(g["ratio_blend"].median())
    gmed = float(r["ratio_blend"].median())
    return dict(exact=m_exact, zc=m_zc, z=m_z, global_=gmed)

def pick_ratio_value(z, c, s, rmaps, default_ratio):
    if rmaps is None: return default_ratio, "default_fixed"
    if (z,c,s) in rmaps["exact"]: return float(rmaps["exact"][(z,c,s)]), "ratio_exact"
    if (z,c) in rmaps["zc"]:      return float(rmaps["zc"][(z,c)]), "ratio_zone_color"
    if z in rmaps["z"]:           return float(rmaps["z"][z]), "ratio_zone"
    if pd.notna(rmaps["global_"]):return float(rmaps["global_"]), "ratio_global"
    return default_ratio, "default_fixed"

def main():
    args = parse_args()
    base = Path(".")

    mi = pd.read_csv(base/args.mi, dtype={"month":"string"})
    mi["coverage_ok"] = mi.get("coverage_ok", False)
    if mi["coverage_ok"].dtype != bool:
        mi["coverage_ok"] = mi["coverage_ok"].astype(str).str.lower().isin(["true","1","yes","si","sí"])
    for c in ["p50","p50_shrunk","wholesale_imputed"]:
        if c not in mi.columns: mi[c] = np.nan
        mi[c] = pd.to_numeric(mi[c], errors="coerce")

    # y preferido
    mi["y"] = np.where(mi["p50_shrunk"].notna(), mi["p50_shrunk"],
                np.where(mi["coverage_ok"] & mi["p50"].notna(), mi["p50"],
                np.where(mi["wholesale_imputed"].notna(), mi["wholesale_imputed"], mi["p50"])))

    # ODEPA retail series por estrato (con fallback mes–zona)
    od = None
    if Path(base/args.odepa).exists():
        odf = pd.read_csv(base/args.odepa, parse_dates=["date"])
        odf["month"] = odf["date"].dt.strftime("%Y-%m")
        odf = odf[["month","zone","egg_color","egg_size","retail_price_clp_per_dozen"]].copy()
        fb = (odf.groupby(["month","zone"])["retail_price_clp_per_dozen"].median()
                .reset_index().rename(columns={"retail_price_clp_per_dozen":"retail_mz"}))
        od = (odf, fb)

    # ratios
    rmaps = None
    if Path(base/args.ratios).exists():
        rmaps = load_ratios_maps(base/args.ratios)
    default_ratio = float(args.default_ratio)

    # seasonal factors (zona)
    sf = None
    if Path(base/args.seasonal).exists():
        sf = pd.read_csv(base/args.seasonal)

    out_rows = []
    for (z, c, s), g in mi.groupby(["zone","egg_color","egg_size"]):
        g = g.sort_values("month")
        months = g["month"].tolist()
        y = g["y"].astype(float).values

        # retail series
        r_series = None
        if od is not None:
            odf, fb = od
            g_od = odf[(odf["zone"]==z) & (odf["egg_color"]==c) & (odf["egg_size"]==s)].copy()
            g_od = g_od.rename(columns={"retail_price_clp_per_dozen":"retail_exact"})
            tmp = pd.DataFrame({"month": months})
            tmp = tmp.merge(g_od[["month","retail_exact"]], on="month", how="left")
            tmp = tmp.merge(fb[fb["zone"]==z], on=["month"], how="left")
            tmp["retail_filled"] = np.where(tmp["retail_exact"].notna(), tmp["retail_exact"], tmp["retail_mz"])
            r_series = tmp["retail_filled"].astype(float).values

        # de-seasonalize by zone factor
        if sf is not None and len(months) >= 13:
            def zfac(m): 
                row = sf[(sf.get("level")=="zone") & (sf.get("zone")==z)]
            y_ds = []
            r_ds = [] if r_series is not None else None
            for i, m in enumerate(months):
                fac_row = sf[(sf.get("level")=="zone") & (sf.get("zone")==z)]
                if not fac_row.empty:
                    f = float(fac_row.iloc[0][f"m{int(m[5:7]):02d}"])
                else:
                    f = 1.0
                y_ds.append(y[i]/f if pd.notna(y[i]) and f not in (0,None) else np.nan)
                if r_series is not None:
                    r_ds.append(r_series[i]/f if pd.notna(r_series[i]) and f not in (0,None) else np.nan)
        else:
            y_ds, r_ds = y, r_series

        method = choose_method(months, y_ds, r_ds, z, c, s, rmaps, default_ratio, alpha=args.alpha)
        q10, q90 = empirical_band(months, y_ds, alpha=args.alpha)

        cur_m, cur_y, cur_r = months[:], list(y_ds), (list(r_ds) if r_ds is not None else None)
        for h in range(1, args.h+1):
            target_m = add_months(months[-1], h)
            r_hat = np.nan
            if method == "retail_ratio" and cur_r is not None:
                r_hat = ensemble_point(cur_m, cur_r, h=1, alpha=args.alpha)
                rr, src = pick_ratio_value(z, c, s, rmaps, default_ratio)
                y_hat_ds = (r_hat * rr) if pd.notna(r_hat) else np.nan
                method_used = f"retail_ratio({src})"
            else:
                y_hat_ds = ensemble_point(cur_m, cur_y, h=1, alpha=args.alpha)
                method_used = "direct"

            # re-seasonalize by zone factor
            if sf is not None:
                fac_row = sf[(sf.get("level")=="zone") & (sf.get("zone")==z)]
                if not fac_row.empty:
                    f = float(fac_row.iloc[0][f"m{int(target_m[5:7]):02d}"])
                else:
                    f = 1.0
                y_hat = y_hat_ds * f if pd.notna(y_hat_ds) else np.nan
            else:
                y_hat = y_hat_ds

            p10 = (y_hat + q10) if pd.notna(y_hat) else np.nan
            p90 = (y_hat + q90) if pd.notna(y_hat) else np.nan

            out_rows.append(dict(month=target_m, zone=z, egg_color=c, egg_size=s,
                                 forecast_docena=y_hat, p10=p10, p90=p90, method=method_used))

            cur_m.append(target_m); cur_y.append(y_hat_ds)
            if cur_r is not None:
                cur_r.append(r_hat)

    out = pd.DataFrame(out_rows).sort_values(["zone","egg_color","egg_size","month"])
    write_outputs(out, base/args.out, excel_locale=args.excel_locale, round_dp=args.round_dp)
    print("OK ->", args.out)

if __name__ == "__main__":
    main()
