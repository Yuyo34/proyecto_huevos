
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A general extractor/cleaner for ODEPA "Precios al Consumidor" (eggs) CSVs.
It reads ALL CSVs from a directory, normalizes to CLP/dozen, aggregates to monthly medians (or means),
and writes a single tidy file usable as input for backfilling wholesale prices.

USAGE EXAMPLES
--------------
python extract_odepa_retail_all.py --raw-dir odepa_raw --out odepa_retail.csv
python extract_odepa_retail_all.py --raw-dir odepa_raw --channels "Feria Libre,Supermercado" --agg median
python extract_odepa_retail_all.py --raw-dir odepa_raw --start 2018-01 --end 2025-12 --agg median

Outputs:
- <out> (default: odepa_retail.csv): date, zone (Región), egg_color, egg_size, retail_price_clp_per_dozen
- odepa_qc_report.csv: counts and basic QC per (month, zone, color, size), plus unit mix and channel coverage.

Notes:
- Handles comma or semicolon separated CSVs and different column header variants.
- Unknown units are skipped and logged in a warnings log next to the output.
- Color/size are parsed from the 'Producto' string; if not found, left as NaN.
"""
import argparse, sys, re, csv, json
import pandas as pd, numpy as np
from pathlib import Path
from calendar import monthrange
from datetime import datetime

def parse_args():
    ap = argparse.ArgumentParser(description="Extract & clean ODEPA egg retail prices to monthly CLP/dozen.")
    ap.add_argument("--raw-dir", type=str, required=True, help="Directory containing ODEPA raw CSVs (by year).")
    ap.add_argument("--out", type=str, default="odepa_retail.csv", help="Output CSV path.")
    ap.add_argument("--qc-out", type=str, default="odepa_qc_report.csv", help="QC report CSV path.")
    ap.add_argument("--channels", type=str, default="Feria Libre,Supermercado",
                    help='Comma-separated list of channels to include (e.g., "Feria Libre" or "Feria Libre,Supermercado").')
    ap.add_argument("--agg", type=str, choices=["median","mean"], default="median", help="Aggregation function across observations per month/region/color/size.")
    ap.add_argument("--start", type=str, default=None, help="Start month YYYY-MM (inclusive).")
    ap.add_argument("--end", type=str, default=None, help="End month YYYY-MM (inclusive).")
    return ap.parse_args()

COLOR_PATTERNS = {
    "blanco": "blanco",
    "blanca": "blanco",
    "color": "color",
    "rojo": "color",  # a veces aparece "huevo rojo"
}
SIZE_PATTERNS = {
    "peque": "S",
    "pequeño": "S",
    "chico": "S",
    "mediano": "M",
    "grande": "L",
    "extra": "XL",
    "super": "XL",
}

def parse_color_size(producto: str):
    if not isinstance(producto, str):
        return (np.nan, np.nan)
    s = producto.lower()
    color = np.nan
    for k, v in COLOR_PATTERNS.items():
        if k in s:
            color = v
            break
    size = np.nan
    for k, v in SIZE_PATTERNS.items():
        if k in s:
            size = v
            break
    return (color, size)

def to_dozen(price, unidad: str):
    if pd.isna(price):
        return np.nan
    u = (unidad or "").strip().lower()
    # Docena explícita
    if "docena" in u:
        return price
    # Bandeja 12 (equivale a docena)
    if "bandeja" in u and "12" in u:
        return price
    # Bandeja 30 -> 30 huevos = 2.5 docenas
    if "bandeja" in u and "30" in u:
        return price / 2.5
    # Unidad/huevo -> multiplicar por 12
    if "unidad" in u or "huevo" in u:
        return price * 12.0
    # Bandeja de 30 unidades (otra variante)
    if "bandeja" in u and "unid" in u and "30" in u:
        return price / 2.5
    return np.nan

def month_end_date(year: int, month: int):
    last_day = monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last_day:02d}"

def read_csv_any(path: Path):
    # Try utf-8, then latin-1; comma then semicolon
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception:
            pass
        try:
            return pd.read_csv(path, sep=";", dtype=str, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"No pude leer {path} con CSV estándar o ';' con utf-8/latin-1.")

def pick_cols(df):
    # Map various header names to canonical
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    return dict(
        anio   = pick("anio","año","ano","a\u00f1o"),
        mes    = pick("mes"),
        region = pick("region","regi\u00f3n"),
        tipo   = pick("tipo de punto monitoreo","tipo_de_punto_monitoreo","tipo_punto","tipo de punto de monitoreo","tipo de punto de monitoreo "),
        producto = pick("producto"),
        unidad   = pick("unidad"),
        precio   = pick("precio promedio","precio_promedio","precio promedio (clp)","precio_promedio_clp")
    )

def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    qc_path  = Path(args.qc_out)
    channels = {s.strip() for s in args.channels.split(",") if s.strip()}

    warn_log = out_path.with_suffix(".warnings.log")
    warnings = []

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        print(f"[X] No hay CSVs en {raw_dir}.")
        sys.exit(1)

    frames = []
    for fp in files:
        try:
            df = read_csv_any(fp)
        except Exception as e:
            warnings.append(f"{fp.name}: no se pudo leer ({e})")
            continue

        cols = pick_cols(df)
        if not all(cols.values()):
            warnings.append(f"{fp.name}: columnas faltantes -> {cols}")
            continue

        # Filter to eggs
        mask_egg = df[cols["producto"]].str.contains("huevo", case=False, na=False)
        df = df[mask_egg].copy()

        # Filter channels
        df[cols["tipo"]] = df[cols["tipo"]].astype(str).str.strip()
        df = df[df[cols["tipo"]].isin(channels)].copy()
        if df.empty:
            continue

        # Coerce numeric price
        # Replace ',' decimal to '.' if needed, safely handling thousands separators
        s = df[cols["precio"]].astype(str).str.replace("\u00a0","", regex=False).str.replace(" ", "", regex=False)
        # Heuristic: remove thousands dots, convert decimal comma to dot
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df[cols["precio"]] = pd.to_numeric(s, errors="coerce")

        # Parse color/size and convert to dozen
        cs = df[cols["producto"]].apply(parse_color_size)
        df["egg_color"] = [x[0] for x in cs]
        df["egg_size"]  = [x[1] for x in cs]
        df["clp_dozen"] = [to_dozen(p, u) for p, u in zip(df[cols["precio"]], df[cols["unidad"]])]

        # Year/month
        df["year"] = pd.to_numeric(df[cols["anio"]], errors="coerce")
        df["month"] = pd.to_numeric(df[cols["mes"]], errors="coerce")

        # Keep relevant rows
        df = df[(df["clp_dozen"].notna()) & (df["clp_dozen"]>0) & df["year"].notna() & df["month"].notna()].copy()

        # Build date at month-end
        df["date"] = [month_end_date(int(y), int(m)) for y, m in zip(df["year"], df["month"])]
        df = df.rename(columns={cols["region"]: "zone", cols["tipo"]: "channel", cols["unidad"]:"unit_raw"})

        frames.append(df[["date","zone","egg_color","egg_size","clp_dozen","channel","unit_raw"]].copy())

    if not frames:
        print("[X] No se generaron filas tras filtrar huevos/canales/unidades.")
        sys.exit(2)

    allx = pd.concat(frames, ignore_index=True)

    # Optional date filtering
    if args.start:
        # convert to YYYY-MM for lexicographic comparison, keep date strings
        allx = allx[allx["date"] >= f"{args.start}-01"]
    if args.end:
        allx = allx[allx["date"] <= f"{args.end}-31"]

    # Aggregate to monthly
    group_keys = ["date","zone","egg_color","egg_size"]
    if args.agg == "median":
        agg = (allx.groupby(group_keys, dropna=False)["clp_dozen"]
                     .median()
                     .reset_index()
                     .rename(columns={"clp_dozen":"retail_price_clp_per_dozen"}))
    else:
        agg = (allx.groupby(group_keys, dropna=False)["clp_dozen"]
                     .mean()
                     .reset_index()
                     .rename(columns={"clp_dozen":"retail_price_clp_per_dozen"}))

    # Sort & save
    agg = agg.sort_values(group_keys)
    agg.to_csv(out_path, index=False)

    # QC report: counts, channels present, unit examples
    qc = (allx.groupby(group_keys, dropna=False)
                .agg(n=("clp_dozen","count"),
                     channels=("channel", lambda s: ",".join(sorted(pd.unique(s))[:5])),
                     unit_examples=("unit_raw", lambda s: ",".join(sorted(pd.unique(s))[:5])))
                .reset_index())
    qc = qc.sort_values(group_keys)
    qc.to_csv(qc_path, index=False)

    # Warnings
    n_missing_cs = agg["egg_color"].isna().sum() + agg["egg_size"].isna().sum()
    warnings_text = []
    if n_missing_cs > 0:
        warnings_text.append(f"Aviso: {n_missing_cs} agregados con color/tamaño NaN (revisa nombres de producto).")
    files_count = len(list(Path(args.raw_dir).glob("*.csv")))
    warnings_text.append(f"Procesados {files_count} archivos desde {args.raw_dir}.")
    if warnings:
        warnings_text.extend(warnings)
    if warnings_text:
        warn_path = Path(args.out).with_suffix(".warnings.log")
        with open(warn_path, "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, warnings_text)))
        print(f"[!] Avisos -> {warn_path}")
    print(f"OK -> {out_path} ; QC -> {qc_path} ; filas={len(agg)}")

if __name__ == "__main__":
    main()
