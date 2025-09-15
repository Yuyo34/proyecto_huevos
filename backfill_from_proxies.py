#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill de precios mayoristas desde retail ODEPA + anclas mayoristas (anchors_wholesale.csv)
Salida: backfilled_wholesale.csv
"""
import pandas as pd, numpy as np
from pathlib import Path

# ----- Parámetros editables -----
DEFAULT_RATIO = 0.85      # mayorista ≈ 85% del retail si no hay anclas suficientes
MIN_MATCHES_FOR_RATIO = 2 # si hay >=2 anclas en un estrato, usar mediana de sus ratios
K_MIN = 3                 # meses mínimos para marcar publish_ok en un estrato
# --------------------------------

BASE = Path(__file__).resolve().parent

def main():
    # 1) Retail ODEPA (mensual por región×color×tamaño)
    odepa_path = BASE / "odepa_retail.csv"
    if not odepa_path.exists():
        print("Falta odepa_retail.csv en la carpeta.")
        return
    odepa = pd.read_csv(odepa_path, parse_dates=["date"])
    keep = ["date","zone","egg_color","egg_size","retail_price_clp_per_dozen"]
    odepa = odepa[keep].dropna()
    odepa["month"] = odepa["date"].dt.strftime("%Y-%m")
    odepa = odepa.drop(columns=["date"])

    # 2) Anchors (opcional pero MUY recomendables)
    anc_path = BASE / "anchors_wholesale.csv"
    anchors = pd.DataFrame(columns=["month","zone","egg_color","egg_size","observed_wholesale_clp_per_dozen","source_notes"])
    if anc_path.exists():
        anchors = pd.read_csv(anc_path, dtype={"month":str})
        # match anchors con retail del mismo mes/estrato para calcular ratio
        merged = anchors.merge(odepa, on=["month","zone","egg_color","egg_size"], how="left")
        merged["ratio"] = merged["observed_wholesale_clp_per_dozen"] / merged["retail_price_clp_per_dozen"]
        ratios = (merged.dropna(subset=["ratio"])
                        .groupby(["zone","egg_color","egg_size"])["ratio"]
                        .agg(list)
                        .reset_index())
    else:
        ratios = pd.DataFrame(columns=["zone","egg_color","egg_size","ratio"])

    # 3) Tabla de ratios por estrato (usa mediana si hay >=2)
    ratio_lut = {}
    for _, row in ratios.iterrows():
        key = (row["zone"], row["egg_color"], row["egg_size"])
        arr = row["ratio"]
        if isinstance(arr, (list, tuple)) and len(arr) >= MIN_MATCHES_FOR_RATIO:
            ratio_lut[key] = float(np.median(arr))
        elif isinstance(arr, (list, tuple)) and len(arr) == 1:
            ratio_lut[key] = float(arr[0])

    # 4) Mayorista backfilled = retail * ratio_estrato (o DEFAULT_RATIO)
    odepa["ratio_used"] = odepa.apply(
        lambda r: ratio_lut.get((r["zone"], r["egg_color"], r["egg_size"]), DEFAULT_RATIO), axis=1
    )
    odepa["wholesale_backfilled"] = odepa["retail_price_clp_per_dozen"] * odepa["ratio_used"]

    # 5) Compras públicas (opcional, para contraste)
    aw_path = BASE / "awards_mercadopublico.csv"
    out = odepa.copy()
    if aw_path.exists():
        aw = pd.read_csv(aw_path, parse_dates=["date"])
        if not aw.empty:
            aw["month"] = aw["date"].dt.strftime("%Y-%m")
            aw_group = (aw.groupby(["month","zone","egg_color","egg_size"])
                          .agg(award_p50=("price_clp","median"),
                               award_n=("price_clp","count"))
                          .reset_index())
            out = out.merge(aw_group, on=["month","zone","egg_color","egg_size"], how="left")

    # 6) publish_ok si el estrato tiene >= K_MIN meses
    counts = out.groupby(["zone","egg_color","egg_size"])["month"].nunique().reset_index(name="months_n")
    out = out.merge(counts, on=["zone","egg_color","egg_size"], how="left")
    out["publish_ok"] = out["months_n"] >= K_MIN

    out = out.sort_values(["zone","egg_color","egg_size","month"])
    out.to_csv(BASE / "backfilled_wholesale.csv", index=False)
    print("OK -> backfilled_wholesale.csv")

if __name__ == "__main__":
    main()
