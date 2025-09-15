#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_month.py
- Lee todos los CSV en inputs/anchors/ (tanto ;/coma decimal como ,/punto).
- Normaliza columnas a: date, zone, egg_color, egg_size, unit, price_clp, source, includes_iva, notes
- Une con anchors_wholesale.csv (maestro), deduplica y guarda.
Uso:
  python ingest_month.py --config config.yml
"""

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

CANON_COLS = ["date","zone","egg_color","egg_size","unit","price_clp","source","includes_iva","notes"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    return ap.parse_args()

def read_csv_auto(p: Path) -> pd.DataFrame:
    # intenta ; con utf-8, ; con latin-1, , con utf-8
    for sep in [";", ","]:
        for enc in ["utf-8-sig","utf-8","latin-1"]:
            try:
                df = pd.read_csv(p, sep=sep, encoding=enc)
                if df.shape[1] >= 3:
                    # normaliza coma decimal si sep es ';'
                    if sep == ";":
                        for c in df.columns:
                            if df[c].dtype == object:
                                def fix_num(x):
                                    if isinstance(x, str) and ("," in x or "." in x):
                                        xs = x.replace(".", "").replace(",", ".")
                                        try:
                                            return float(xs)
                                        except:
                                            return x
                                    return x
                                df[c] = df[c].map(fix_num)
                    return df
            except Exception:
                continue
    return pd.read_csv(p)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    alias = {
        "fecha":"date","region":"zone","zona":"zone",
        "color":"egg_color","tamaño":"egg_size","tamano":"egg_size","size":"egg_size",
        "unidad":"unit","precio":"price_clp","precio_clp":"price_clp","precio_docena":"price_clp",
        "fuente":"source","iva":"includes_iva","notas":"notes"
    }
    out = df.copy()
    out.columns = [alias.get(str(c).strip().lower(), str(c).strip()) for c in out.columns]
    for c in CANON_COLS:
        if c not in out.columns: out[c] = np.nan
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["zone"] = out["zone"].astype(str).str.strip()
    out["egg_color"] = out["egg_color"].astype(str).str.strip().str.lower()
    out["egg_size"] = out["egg_size"].astype(str).str.strip().str.upper()
    out["unit"] = out["unit"].astype(str).str.strip().str.lower()
    out["price_clp"] = pd.to_numeric(out["price_clp"], errors="coerce")
    out["includes_iva"] = out["includes_iva"].map(lambda x: str(x).lower() if pd.notna(x) else x)
    return out[CANON_COLS]

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    anchors_dir = Path(cfg["paths"]["inputs_anchors_dir"]).expanduser()
    anchors_dir.mkdir(parents=True, exist_ok=True)
    master_path = Path(cfg["paths"]["anchors_master"]).expanduser()

    frames = []
    found = False
    for p in anchors_dir.glob("*.csv"):
        if p.name.startswith("~$"):  # temporales de Excel
            continue
        df = read_csv_auto(p)
        df = normalize_cols(df)
        if df["date"].notna().sum() == 0:
            print(f"[WARN] {p.name}: no hay fechas válidas, omitido.")
            continue
        frames.append(df)
        found = True
        print(f"[OK] leído {p.name}: {len(df)} filas")

    if not found:
        print("[INFO] No se encontraron nuevos CSV en inputs/anchors/. Nada que ingerir.")
        return

    new_df = pd.concat(frames, ignore_index=True)
    if master_path.exists():
        master = pd.read_csv(master_path, parse_dates=["date"])
        master = normalize_cols(master)
    else:
        master = pd.DataFrame(columns=CANON_COLS)

    all_df = pd.concat([master, new_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["date","zone","egg_color","egg_size","unit","price_clp","source"]).sort_values(["date","zone","egg_color","egg_size"]).reset_index(drop=True)
    all_df.to_csv(master_path, index=False, encoding="utf-8")

    if cfg["params"].get("excel_locale", False):
        df2 = all_df.copy()
        for c in df2.select_dtypes(include=[float,int]).columns:
            df2[c] = df2[c].map(lambda x: ("" if pd.isna(x) else str(x).replace(".", ",")))
        all_df_path_excel = Path(str(master_path).replace(".csv","_excel.csv"))
        df2.to_csv(all_df_path_excel, sep=";", index=False, encoding="utf-8-sig")

    added = len(all_df) - (len(master) if isinstance(master, pd.DataFrame) else 0)
    print(f"[OK] Maestro actualizado: {master_path} (+{added} filas nuevas)")

if __name__ == "__main__":
    main()
