#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_wholesale_index.py (ratios + pooling + monotonicidad + CSV amigable Excel)
- Usa ratios_calibrated.csv (ratio_blend) con fallback jerárquico para imputar.
- Winsor por estrato/mes.
- coverage_ok por K_MIN.
- Pooling jerárquico: si n<k_pool, contrae p50 hacia p50_zona_color.
- Ajuste de monotonicidad por tamaño (XL≥L≥M≥S) dentro de cada color.
- Exporta CSV estándar y, si se pide, versión Excel (sep=';' y coma decimal).

Uso:
  python build_wholesale_index.py --raw raw_prices_clean.csv --odepa odepa_retail.csv --ratios ratios_calibrated.csv --excel-locale --round-dp 0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- Excel helpers ----------------
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

# ---------------- Config ----------------
IVA_RATE = 0.19
PREFER_NET = True
K_MIN = 5
K_POOL = 6
WINSOR_P = 0.025
DEFAULT_RATIO = 0.85
SIZE_ORDER = {"S":0,"M":1,"L":2,"XL":3}

ZONE_MAP = {
    "Region Metropolitana": "Región Metropolitana de Santiago",
    "Región Metropolitana": "Región Metropolitana de Santiago",
    "Metropolitana": "Región Metropolitana de Santiago",
    "Region de Valparaiso": "Región de Valparaíso",
    "Región de Valparaiso": "Región de Valparaíso",
}
SIZE_MAP = {"ESPECIAL":"XL","EXTRA":"L","GRANDE":"M","MEDIANO":"S","XL":"XL","L":"L","M":"M","S":"S"}
COLOR_MAP = {"blanco":"blanco","white":"blanco","color":"color","rojo":"color"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw_prices_clean.csv")
    ap.add_argument("--odepa", default="odepa_retail.csv")
    ap.add_argument("--ratios", default="ratios_calibrated.csv")
    ap.add_argument("--kmin", type=int, default=K_MIN)
    ap.add_argument("--kpool", type=int, default=K_POOL)
    ap.add_argument("--winsor", type=float, default=WINSOR_P)
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    ap.add_argument("--prefer-net", action="store_true", default=PREFER_NET)
    ap.add_argument("--out1", default="monthly_index.csv")
    ap.add_argument("--out2", default="monthly_index_with_proxies.csv")
    ap.add_argument("--excel-locale", action="store_true", default=False)
    ap.add_argument("--round-dp", type=int, default=0)
    return ap.parse_args()

def read_csv_auto(path: Path):
    if not path.exists(): return None
    for sep in [None, ",", ";"]:
        try:
            df = pd.read_csv(path, sep=None if sep is None else sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python")

def to_dozen(price, unit):
    if pd.isna(price): return np.nan
    u = (str(unit) or "").strip().lower()
    if "docena" in u: return price
    if "bandeja" in u and "12" in u: return price
    if "bandeja" in u and "30" in u: return price / 2.5
    if "unidad" in u or "huevo" in u: return price * 12.0
    if u in ("docena","dozen","doz"): return price
    if u in ("huevo","unit","egg"):   return price * 12.0
    if u in ("bandeja_30","bandeja","tray_30"): return price / 2.5
    if u in ("caja_100","box_100"):   return price / (100.0/12.0)
    if u in ("caja_180","box_180"):   return price / 15.0
    if u in ("caja_360","box_360"):   return price / 30.0
    return np.nan

def to_net(price, includes_iva, prefer_net=True):
    if pd.isna(price): return np.nan
    if not prefer_net: return price
    s = str(includes_iva).strip().lower()
    return price/(1.0+IVA_RATE) if s in ("true","1","yes","si","sí") else price

def winsorize(s: pd.Series, p=0.025):
    if s.isna().all(): return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "zone" in out:
        out["zone"] = out["zone"].astype(str).str.strip().replace(ZONE_MAP)
    if "egg_color" in out:
        out["egg_color"] = out["egg_color"].astype(str).str.strip().str.lower()
        out["egg_color"] = out["egg_color"].map(lambda x: COLOR_MAP.get(x, x))
    if "egg_size" in out:
        out["egg_size"] = out["egg_size"].astype(str).str.strip().str.upper()
        out["egg_size"] = out["egg_size"].map(lambda x: SIZE_MAP.get(x, x))
    return out

def load_ratios(path: Path):
    if not path.exists(): return None
    r = read_csv_auto(path)
    if r is None or r.empty or "ratio_blend" not in r.columns: return None
    r_exact = r[["zone","egg_color","egg_size","ratio_blend"]].dropna(subset=["ratio_blend"])
    r_zc = (r_exact.groupby(["zone","egg_color"])["ratio_blend"].median().reset_index()
                 .rename(columns={"ratio_blend":"ratio_zc"}))
    r_z  = (r_exact.groupby(["zone"])["ratio_blend"].median().reset_index()
                 .rename(columns={"ratio_blend":"ratio_z"}))
    r_g  = float(r_exact["ratio_blend"].median()) if not r_exact.empty else np.nan
    return dict(r_exact=r_exact, r_zc=r_zc, r_z=r_z, r_global=r_g)

def pick_ratio(row, ratios, default_ratio):
    z, c, s = row["zone"], row["egg_color"], row["egg_size"]
    if ratios is None: return default_ratio, "default_fixed"
    rex = ratios["r_exact"]
    m = rex[(rex["zone"]==z) & (rex["egg_color"]==c) & (rex["egg_size"]==s)]
    if not m.empty and pd.notna(m["ratio_blend"].iloc[0]): return float(m["ratio_blend"].iloc[0]), "ratio_exact"
    m2 = ratios["r_zc"][(ratios["r_zc"]["zone"]==z) & (ratios["r_zc"]["egg_color"]==c)]
    if not m2.empty and pd.notna(m2["ratio_zc"].iloc[0]): return float(m2["ratio_zc"].iloc[0]), "ratio_zone_color"
    m3 = ratios["r_z"][ratios["r_z"]["zone"]==z]
    if not m3.empty and pd.notna(m3["ratio_z"].iloc[0]): return float(m3["ratio_z"].iloc[0]), "ratio_zone"
    rg = ratios.get("r_global", np.nan)
    if pd.notna(rg): return float(rg), "ratio_global"
    return default_ratio, "default_fixed"

def monotone_adjust(group_df, value_col="p50_shrunk"):
    d = group_df.copy().sort_values("egg_size", key=lambda s: s.map(SIZE_ORDER).fillna(99), ascending=False)
    v = d[value_col].values.astype(float)
    v_adj = np.maximum.accumulate(v)  # asegura XL>=L>=M>=S
    d[value_col] = v_adj
    return d.sort_values("egg_size", key=lambda s: s.map(SIZE_ORDER).fillna(99))

def main():
    args = parse_args()
    base = Path(__file__).resolve().parent

    # RAW
    raw = read_csv_auto(base/args.raw)
    if raw is None or raw.empty:
        print(f"[ERROR] Falta o vacío: {args.raw}"); return
    need = {"date","zone","egg_color","egg_size","unit","price_clp"}
    if not need.issubset(set(raw.columns)):
        print(f"[ERROR] {args.raw} incompleto. Falta alguna de: {need}"); return

    df = raw.copy()
    df["price_clp"] = pd.to_numeric(df["price_clp"], errors="coerce")
    df["price_clp"] = df.apply(lambda r: to_net(r["price_clp"], r.get("includes_iva"), args.prefer_net), axis=1)
    df["price_per_dozen"] = df.apply(lambda r: to_dozen(r["price_clp"], r.get("unit")), axis=1)
    df = df[df["price_per_dozen"].notna() & (df["price_per_dozen"]>0)].copy()
    df = normalize_keys(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.strftime("%Y-%m")

    keys = ["month","zone","egg_color","egg_size"]
    df["price_w"] = df.groupby(keys)["price_per_dozen"].transform(lambda s: winsorize(s, args.winsor))

    agg = (df.groupby(keys)
             .agg(n=("price_w","count"),
                  p50=("price_w","median"),
                  p10=("price_w", lambda s: s.quantile(0.10)),
                  p90=("price_w", lambda s: s.quantile(0.90)))
             .reset_index())
    agg["coverage_ok"] = agg["n"] >= args.kmin

    # Pooling jerárquico hacia p50_zona_color cuando n<kpool
    zc = (agg.groupby(["month","zone","egg_color"])["p50"].median()
             .reset_index().rename(columns={"p50":"p50_zona_color"}))
    out = agg.merge(zc, on=["month","zone","egg_color"], how="left")
    lam = (out["n"] / (out["n"] + float(args.kpool))).clip(lower=0.0, upper=1.0)
    out["p50_shrunk"] = lam*out["p50"] + (1.0-lam)*out["p50_zona_color"]

    # Monotonicidad por color dentro de cada month×zone
    out = (out.groupby(["month","zone","egg_color"], group_keys=False)
              .apply(lambda g: monotone_adjust(g, value_col="p50_shrunk")))

    out_sorted = out.sort_values(keys)
    write_outputs(out_sorted, base/args.out1, excel_locale=args.excel_locale, round_dp=args.round_dp)

    # ODEPA retail + imputado con ratios calibrados
    od = read_csv_auto(base/args.odepa)
    if od is None or od.empty or "date" not in od.columns or "retail_price_clp_per_dozen" not in od.columns:
        print(f"[WARN] No pude cargar {args.odepa}. Guardo {args.out2} sin imputado.")
        write_outputs(out_sorted, base/args.out2, excel_locale=args.excel_locale, round_dp=args.round_dp)
        print("OK ->", args.out1, ",", args.out2); return

    od = normalize_keys(od)
    od["date"] = pd.to_datetime(od["date"], errors="coerce")
    od["month"] = od["date"].dt.strftime("%Y-%m")
    od["retail_price_clp_per_dozen"] = pd.to_numeric(od["retail_price_clp_per_dozen"], errors="coerce")
    orw = od[["month","zone","egg_color","egg_size","retail_price_clp_per_dozen"]].copy()
    fb = (od.groupby(["month","zone"])["retail_price_clp_per_dozen"]
            .median().reset_index().rename(columns={"retail_price_clp_per_dozen":"retail_mz"}))

    out2 = out_sorted.merge(orw, on=keys, how="left").merge(fb, on=["month","zone"], how="left")
    out2["retail_filled"] = np.where(out2["retail_price_clp_per_dozen"].notna(),
                                     out2["retail_price_clp_per_dozen"], out2["retail_mz"])

    # ratios calibrados
    ratios = load_ratios(base/args.ratios)
    default_ratio = float(args.ratio)
    ratios_used, sources = [], []
    for _, r in out2.iterrows():
        rr, src = pick_ratio(r, ratios, default_ratio)
        ratios_used.append(rr); sources.append(src)
    out2["ratio_used"] = ratios_used
    out2["ratio_used_source"] = sources
    out2["wholesale_imputed"] = out2["retail_filled"] * out2["ratio_used"]

    out2["delta_vs_imputed_pct"] = np.where(
        out2[["p50_shrunk","wholesale_imputed"]].notna().all(axis=1),
        100.0*(out2["p50_shrunk"] - out2["wholesale_imputed"]) / out2["wholesale_imputed"],
        np.nan
    )
    out2 = out2.sort_values(keys)
    write_outputs(out2, base/args.out2, excel_locale=args.excel_locale, round_dp=args.round_dp)
    print("OK ->", args.out1, ",", args.out2)

if __name__ == "__main__":
    main()
