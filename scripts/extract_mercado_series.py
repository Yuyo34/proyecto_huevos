import re, pdfplumber, pandas as pd
from pathlib import Path
from unidecode import unidecode

PDF = "data/Boletin_Chilehuevos_Ago_2025.pdf"  # ajusta si renombraste
OUT = Path("data"); OUT.mkdir(exist_ok=True)

MONTHS = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"sept":9,"sep":9,"oct":10,"nov":11,"dic":12}
ROW_KEYS = ["produccion","importacion","% importados","total"]

def normnum(x):
    if pd.isna(x): return None
    s = str(x).strip().replace(".","").replace(",",".")
    return float(s) if re.fullmatch(r"-?\d+(?:\.\d+)?", s) else None

def detect_mercado_table():
    with pdfplumber.open(PDF) as pdf:
        for page in pdf.pages:
            txt = (page.extract_text() or "").lower()
            if "mercado del huevo" in unidecode(txt):
                for tbl in page.extract_tables() or []:
                    df = pd.DataFrame(tbl)
                    if df.shape[0] >= 4 and df.shape[1] >= 6:
                        yield df

def tidy(df):
    # promover primera fila si parece header
    header = df.iloc[0].astype(str).str.strip()
    if (header.str.contains("Mes", case=False).any()) or (header.str.fullmatch(r"\d{4}").fillna(False).sum()>=3):
        df.columns = header
        df = df.iloc[1:]

    # primera col: indicador (producción, importación, % importados, total)
    df = df.rename(columns={df.columns[0]: "Indicador"})
    df["Indicador"] = df["Indicador"].astype(str).str.strip()
    # columnas de año (4 dígitos)
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]
    if not year_cols:
        # a veces la tabla tiene meses como columnas; en ese caso abortamos
        return None

    # filas válidas (contengan alguno de los keys)
    keep = []
    for _, row in df.iterrows():
        t = unidecode(str(row["Indicador"])).lower()
        if any(k in t for k in ["produccion","importacion","% importados","importados","total"]):
            keep.append(row)
    if not keep: return None
    df = pd.DataFrame(keep)

    # ahora tenemos filas = indicadores y columnas = años con valores por MES en subcolumnas?
    # en muchos boletines, cada "celda" es otra tabla anidada; si no: el formato típico es "Mes" en segunda col
    # Intentaremos caso más común: segunda col es "Mes"
    # Si no existe, intentamos que la primera col sean meses y la segunda el primer año (y así).
    mes_col = None
    for c in df.columns:
        if unidecode(str(c)).lower().startswith("mes"):
            mes_col = c; break
    if mes_col is None:
        # si no hay columna Mes, asumimos que las filas posteriores contienen meses en la primera col y los indicadores en otra forma -> fallback
        # Como fallback: devolvemos None para forzar seguir probando otras tablas detectadas
        return None

    # expandir a largo: para cada indicador, meses x años
    records = []
    for _, row in df.iterrows():
        ind = str(row["Indicador"]).strip()
        for y in year_cols:
            for mes, val in zip(df[mes_col], df[y]):
                if pd.isna(mes): continue
                mtxt = unidecode(str(mes)).lower().strip().replace("set","sept")
                m = None
                for k,v in MONTHS.items():
                    if k in mtxt:
                        m = v; break
                if m is None: continue
                v = normnum(val)
                if v is None: continue
                date = pd.Timestamp(int(y), m, 1)
                records.append((ind, date, v))
    if not records: return None
    out = pd.DataFrame(records, columns=["indicator","date","value"]).sort_values(["indicator","date"])
    return out

# buscar la tabla válida
out = None
for df in detect_mercado_table():
    out = tidy(df)
    if out is not None:
        break

if out is None or out.empty:
    raise SystemExit("No se pudo estructurar la tabla 'Mercado del Huevo'. Abre el PDF con Excel/PowerQuery o Tabula y exporta esa tabla.")

# normalizar nombres de indicador
def norm_ind(s):
    t = unidecode(str(s)).lower()
    if "importac" in t and "%" in t:
        return "% importados"
    if "importac" in t:
        return "Importacion"
    if "producc" in t:
        return "Produccion"
    if "total" in t:
        return "Total"
    return s

out["indicator"] = out["indicator"].map(norm_ind)

# guardar 4 series separadas
for ind in ["Produccion","Importacion","% importados","Total"]:
    s = out[out["indicator"]==ind][["date","value"]].dropna()
    if not s.empty:
        s.to_csv(OUT / f"mercado_{ind.replace('% ','pct_').replace(' ','_').lower()}.csv", index=False)
        print(f"OK -> {OUT / f'mercado_{ind.replace('% ','pct_').replace(' ','_').lower()}.csv'} ({len(s)} filas)")
