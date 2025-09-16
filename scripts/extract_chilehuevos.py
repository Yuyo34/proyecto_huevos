import argparse, re
import pdfplumber, pandas as pd
from pathlib import Path
from unidecode import unidecode

TABLE_HINTS = {
    "pollitas": "VENTA DE POLLITAS",
    "produccion": "ESTIMACIÓN DE PRODUCCIÓN DE HUEVOS",
    "ciclo": "ESTIMACIÓN DE PRODUCCIÓN DE HUEVOS POR CICLO",
    "mercado": "MERCADO DEL HUEVO",
    "usdclp": "DÓLAR OBSERVADO",
}

MONTHS = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"sept":9,"sep":9,"oct":10,"nov":11,"dic":12}

def normnum(x):
    if pd.isna(x): return None
    s = str(x).strip().replace('.','').replace(',','.')
    return float(s) if re.fullmatch(r'-?\d+(?:\.\d+)?', s) else None

def wide_to_long(df):
    # busca columna Mes y columnas Año (4 dígitos)
    cmes = None
    for c in df.columns:
        u = unidecode(str(c)).lower()
        if u.startswith("mes"): cmes = c; break
    if cmes is None:
        # heurística: primera col
        cmes = df.columns[0]
    year_cols = [c for c in df.columns if re.fullmatch(r'\d{4}', str(c).strip())]
    out = []
    for _, row in df.iterrows():
        mes_txt = unidecode(str(row[cmes])).lower().strip()
        mes_txt = mes_txt.replace('set','sept')
        m = next((MONTHS[k] for k in MONTHS if k in mes_txt), None)
        if not m: continue
        for y in year_cols:
            v = normnum(row[y])
            if v is None: continue
            out.append((pd.Timestamp(int(y), m, 1), float(v)))
    return pd.DataFrame(out, columns=["date","value"]).sort_values("date")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True, help="Ruta al PDF del boletín")
    ap.add_argument("-o","--outdir", default="data", help="Carpeta de salida para CSVs")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    raw = {k: [] for k in TABLE_HINTS}

    with pdfplumber.open(args.input) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for key, label in TABLE_HINTS.items():
                if label.split()[0] in text:
                    for tbl in page.extract_tables() or []:
                        df = pd.DataFrame(tbl)
                        if df.shape[0] > 1:
                            header = df.iloc[0].astype(str).str.strip()
                            # si el header parece años/“Mes”, úsalo
                            if (header.str.fullmatch(r"\d{4}").fillna(False).sum()>=3) or header.str.contains("Mes", case=False).any():
                                df.columns = header
                                df = df.iloc[1:]
                        raw[key].append(df)

    def pick_best(dfs):
        if not dfs: return None
        def score(df):
            years = [c for c in df.columns if re.fullmatch(r'\d{4}', str(c).strip())]
            return len(years)*1000 + len(df)
        return max(dfs, key=score)

    # Guardar tablas principales en formato largo (date,value)
    for key in ["pollitas","produccion","mercado","usdclp"]:
        best = pick_best(raw.get(key, []))
        if best is not None and not best.empty:
            longdf = wide_to_long(best)
            longdf.to_csv(outdir / f"chilehuevos_{key}.csv", index=False)
            print(f"OK -> {outdir / f'chilehuevos_{key}.csv'} ({len(longdf)} filas)")
        else:
            print(f"AVISO: no se pudo extraer tabla '{key}'")

    # Ciclo: guardar crudo también por si requiere tratamiento distinto
    best_ciclo = pick_best(raw.get("ciclo", []))
    if best_ciclo is not None and not best_ciclo.empty:
        best_ciclo.to_csv(outdir / "chilehuevos_ciclo_raw.csv", index=False)
        print(f"OK -> {outdir / 'chilehuevos_ciclo_raw.csv'} (crudo)")

if __name__ == "__main__":
    main()
