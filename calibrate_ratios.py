#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_ratios.py (robusto a choque de columnas + CSV amigable Excel)
Aprende el ratio retail->mayorista por estrato con EWMA + backtesting y shrinkage.
"""

import argparse, json
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
    ap.add_argument("--out", default="ratios_calibrated.csv")
    ap.add_argument("--alphas", default="0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--blend_k", type=float, default=5.0)
    ap.add_argument("--metric", choices=["mdape","mape"], default="mdape")
    ap.add_argument("--excel-locale", action="store_true", default=False)
    ap.add_argument("--round-dp", type=int, default=4)
    return ap.parse_args()

def mdape(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    mask = (np.isfinite(y) & np.isfinite(yhat) & (y!=0))
    if not mask.any(): return np.nan
    return np.median(np.abs((yhat[mask]-y[mask]) / y[mask]))*100.0

def mape(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    mask = (np.isfinite(y) & np.isfinite(yhat) & (y!=0))
    if not mask.any(): return np.nan
    return np.mean(np.abs((yhat[mask]-y[mask]) / y[mask]))*100.0

def ewma(series, alpha=0.3):
    arr = np.array([x for x in series if pd.notna(x)], float)
    if arr.size == 0: return np.nan
    level = arr[0]
    for x in arr[1:]:
        level = alpha*x + (1-alpha)*level
    return float(level)

def main():
    args = parse_args()
    base = Path(".")

    # --- Carga ---
    mi = pd.read_csv(base/args.mi, dtype={"month":"string"})
    od = pd.read_csv(base/args.odepa, parse_dates=["date"])
    od["month"] = od["date"].dt.strftime("%Y-%m")

    keys = ["month","zone","egg_color","egg_size"]

    # Serie objetivo y
    mi["coverage_ok"] = mi.get("coverage_ok", False)
    if mi["coverage_ok"].dtype != bool:
        mi["coverage_ok"] = mi["coverage_ok"].astype(str).str.lower().isin(["true","1","yes","si","sí"])
    for c in ["p50","wholesale_imputed"]:
        if c not in mi.columns: mi[c] = np.nan
        mi[c] = pd.to_numeric(mi[c], errors="coerce")
    mi["y"] = np.where(mi["coverage_ok"] & mi["p50"].notna(), mi["p50"],
                np.where(mi["wholesale_imputed"].notna(), mi["wholesale_imputed"], mi["p50"]))

    # ODEPA con nombre único
    od = od.rename(columns={"retail_price_clp_per_dozen":"retail_odepa"})
    need = {"month","zone","egg_color","egg_size","retail_odepa"}
    if not need.issubset(od.columns):
        raise RuntimeError(f"odepa_retail.csv sin columnas requeridas (faltan: {need - set(od.columns)})")

    # Merge exacto
    df = mi.merge(od[list(need)], on=keys, how="left", validate="m:1")

    # Fallback mes–zona en columna única (sin chocar nombres del MI)
    fb = (od.groupby(["month","zone"])["retail_odepa"]
            .median().reset_index().rename(columns={"retail_odepa":"retail_mz_fb"}))
    df = df.merge(fb, on=["month","zone"], how="left")

    # Si el MI ya traía retail, úsalo como retail_exact (candidatos seguros)
    retail_exact = None
    for cand in ["retail_price_clp_per_dozen", "retail_filled", "retail_mz"]:
        if cand in df.columns:
            retail_exact = cand
            break

    # retail_base = preferir retail_exact (si viene en MI), si no retail_odepa (de ODEPA)
    df["retail_base"] = np.where(
        pd.notna(df[retail_exact]) if retail_exact else False,
        pd.to_numeric(df[retail_exact], errors="coerce"),
        pd.to_numeric(df["retail_odepa"], errors="coerce")
    )

    # retail_filled = retail_base si existe; si no, usar retail_mz_fb
    df["retail_filled"] = np.where(df["retail_base"].notna(), df["retail_base"], df["retail_mz_fb"])

    # Ratio observado solo con p50 y retail>0
    df["ratio_obs"] = np.where(
        df["p50"].notna() & pd.to_numeric(df["retail_filled"], errors="coerce").gt(0),
        df["p50"] / pd.to_numeric(df["retail_filled"], errors="coerce"),
        np.nan
    )

    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    for (zone, col, size), g in df.groupby(["zone","egg_color","egg_size"]):
        g = g.sort_values("month")
        r = g["ratio_obs"]
        ratio_global = np.nanmedian(r.values) if r.notna().any() else np.nan

        if r.notna().sum() < 3:
            rows.append(dict(zone=zone, egg_color=col, egg_size=size,
                             alpha_opt=np.nan, metric=args.metric, score=np.nan,
                             n_obs=int(r.notna().sum()), ratio_global=ratio_global,
                             ratio_ewma=np.nan, ratio_blend=ratio_global))
            continue

        # Backtest 1-paso sobre y = retail_filled * ratio_hat
        scores = []
        for alpha in alphas:
            preds, trues = [], []
            for i in range(2, len(g)-1):
                hist = r.iloc[:i+1].dropna()
                ratio_hat = ewma(hist.values, alpha=alpha)
                if not np.isfinite(ratio_hat): continue
                retail_next = g["retail_filled"].iloc[i+1]
                yhat = retail_next * ratio_hat if pd.notna(retail_next) else np.nan
                ytrue = g["y"].iloc[i+1]
                if pd.notna(yhat) and pd.notna(ytrue) and ytrue != 0:
                    preds.append(yhat); trues.append(ytrue)
            if len(preds) >= 3:
                score = mdape(trues, preds) if args.metric=="mdape" else mape(trues, preds)
                scores.append((alpha, score))
        if scores:
            alpha_opt, best_score = sorted(scores, key=lambda x: x[1])[0]
        else:
            alpha_opt, best_score = np.nan, np.nan

        use_alpha = alpha_opt if np.isfinite(alpha_opt) else 0.35
        ratio_ewma = ewma(r.dropna().values, alpha=use_alpha)

        n = int(r.notna().sum())
        w = n/(n+float(args.blend_k)) if np.isfinite(n) else 0.0
        if np.isfinite(ratio_ewma) and np.isfinite(ratio_global):
            ratio_blend = w*ratio_ewma + (1-w)*ratio_global
        elif np.isfinite(ratio_ewma):
            ratio_blend = ratio_ewma
        else:
            ratio_blend = ratio_global

        rows.append(dict(zone=zone, egg_color=col, egg_size=size,
                         alpha_opt=use_alpha, metric=args.metric, score=best_score,
                         n_obs=n, ratio_global=ratio_global, ratio_ewma=ratio_ewma, ratio_blend=ratio_blend))

    out = pd.DataFrame(rows).sort_values(["zone","egg_color","egg_size"])
    write_outputs(out, base/args.out, excel_locale=args.excel_locale, round_dp=args.round_dp)

    summary = {
        "rows": int(len(out)),
        "alphas": alphas,
        "metric": args.metric,
        "median_score": float(np.nanmedian(out["score"])) if "score" in out else None,
        "estratos_con_ratio": int(out["n_obs"].gt(0).sum()),
        "estratos_backtesteados": int(out["n_obs"].ge(3).sum())
    }
    Path("ratios_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK ->", args.out, "| resumen -> ratios_summary.json")

if __name__ == "__main__":
    main()
