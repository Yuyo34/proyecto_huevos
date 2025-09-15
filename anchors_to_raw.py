#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anchors_to_raw.py
Convierte tus anclas mayoristas (CLP/docena netos) en raw_prices.csv
Acepta:
  - anchors_wholesale.csv (coma)  o
  - anchors_wholesale_semicolon.csv (";")

Hace:
  1) Lee y normaliza columnas (month/zone/egg_color/egg_size/precio).
  2) Mapea tamaños ESPECIAL/EXTRA/GRANDE/MEDIANO -> XL/L/M/S.
  3) Normaliza nombres de zona para que calcen con ODEPA.
  4) Arreglo de magnitud si detecta valores 10× (> 6000 por docena).
  5) Escribe raw_prices.csv (unidad=docena, neto).
"""

import sys
from calendar import monthrange
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

# Zonas típicas a mapear hacia los nombres que usa ODEPA
ZONE_MAP = {
    "Region Metropolitana": "Región Metropolitana de Santiago",
    "Metropolitana": "Región Metropolitana de Santiago",
    "Región Metropolitana": "Región Metropolitana de Santiago",
    "Region de Valparaiso": "Región de Valparaíso",
    "Región de Valparaiso": "Región de Valparaíso",
}

SIZE_MAP = {"ESPECIAL":"XL","EXTRA":"L","GRANDE":"M","MEDIANO":"S",
            "XL":"XL","L":"L","M":"M","S":"S"}

# --- utilidades ---
def month_to_date(m):
    y, mo = str(m).split("-")
    last = monthrange(int(y), int(mo))[1]
    return f"{y}-{mo}-{last:02d}"

def parse_number(x):
    s = str(x).replace("$","").replace("\u00a0","").replace(" ","")
    s = s.replace(".","").replace(",",".")
    try:
        return float(s)
    except Exception:
        return None

def try_read_csv(path: Path):
    """Intenta leer con autodetección; si trae 1 sola columna, reintenta con ';'."""
    # 1) intento estándar (coma)
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # 2) intento con ';'
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # 3) intento con engine=python (autodetect)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    raise ValueError(f"No pude parsear correctamente {path.name} (delimitadores).")

def load_anchors():
    candidates = [
        BASE / "anchors_wholesale.csv",
        BASE / "anchors_wholesale_semicolon.csv",
    ]
    for p in candidates:
        if p.exists():
            return try_read_csv(p), p.name
    raise FileNotFoundError("No encontré anchors_wholesale.csv ni anchors_wholesale_semicolon.csv en la carpeta.")

def normalize_columns(df):
    """Mapea encabezados variados a nombres canónicos."""
    # índice por nombre en minúsculas sin espacios
    norm = {c.strip().lower(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in norm: return norm[o]
        return None

    col_month = pick("month","mes","periodo","fecha")
    col_zone  = pick("zone","zona","region","región")
    col_color = pick("egg_color","color","huevo_color")
    col_size  = pick("egg_size","size","tamaño","tamano")
    col_price = pick("observed_wholesale_clp_per_dozen","precio_docena_clp","precio_clp_docena","precio_clp")
    if not all([col_month, col_zone, col_color, col_size, col_price]):
        raise ValueError(f"Faltan columnas requeridas en anchors: tengo {list(df.columns)}")

    out = pd.DataFrame()
    out["month"] = df[col_month].astype(str).str.strip()
    out["zone"]  = df[col_zone].astype(str).str.strip()
    out["egg_color"] = df[col_color].astype(str).str.strip().str.lower()
    out["egg_size"]  = df[col_size].astype(str).str.strip().str.upper()
    out["observed_wholesale_clp_per_dozen"] = df[col_price].apply(parse_number)
    # Normalizaciones
    out["egg_size"] = out["egg_size"].map(lambda s: SIZE_MAP.get(s, s))
    out["zone"] = out["zone"].replace(ZONE_MAP)
    return out

def main():
    try:
        df_raw, fname = load_anchors()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    df = normalize_columns(df_raw)

    # Guard 10× (si varias docenas superan 6000, suelen venir multiplicadas x10)
    x = pd.to_numeric(df["observed_wholesale_clp_per_dozen"], errors="coerce")
    mask = x > 6000
    if mask.any():
        df.loc[mask, "observed_wholesale_clp_per_dozen"] = x[mask] / 10.0
        print(f"[WARN] Corregidas {int(mask.sum())} filas con magnitud sospechosa (>6000) ÷10.")

    # Filtrar filas válidas
    df = df.dropna(subset=["month","zone","egg_color","egg_size","observed_wholesale_clp_per_dozen"])
    if df.empty:
        print("[ERROR] No quedaron filas válidas en las anclas.")
        sys.exit(2)

    # Construir RAW (unidad = docena, neto)
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(
            date = month_to_date(r["month"]),
            zone = r["zone"],
            egg_color = r["egg_color"],
            egg_size = r["egg_size"],
            source_name = "anchor",
            source_url = "",
            unit = "docena",
            price_clp = float(r["observed_wholesale_clp_per_dozen"]),
            includes_iva = False,
            includes_freight = False,
            freight_clp = "",
            is_promo = False,
            notes = f"from anchors ({fname})"
        ))
    raw = pd.DataFrame(rows, columns=[
        "date","zone","egg_color","egg_size","source_name","source_url",
        "unit","price_clp","includes_iva","includes_freight","freight_clp",
        "is_promo","notes"
    ])
    out_path = BASE / "raw_prices.csv"
    raw.to_csv(out_path, index=False, encoding="utf-8")
    print(f"OK -> {out_path} ({len(raw)} filas)")

if __name__ == "__main__":
    main()
