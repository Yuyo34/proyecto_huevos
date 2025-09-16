import re, pdfplumber, pandas as pd
from pathlib import Path
from unidecode import unidecode

PDF = r"data\Boletin_Chilehuevos_Ago_2025.pdf"  # ajusta el nombre si difiere
OUT = Path("data"); OUT.mkdir(exist_ok=True)

MONTHS = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"set":9,"sep":9,"oct":10,"nov":11,"dic":12}

def parse_month(s):
    s = (s or "").strip().lower().replace("sept","sep")
    m = re.match(r"([a-záéíóú]{3})[-\s]?(\d{2}|\d{4})", s)
    if not m: return None
    mon = m.group(1)[:3]; yr = int(m.group(2))
    mon = MONTHS.get(mon); 
    if not mon: return None
    if yr < 100: yr += 2000 if yr <= 50 else 1900
    return pd.Timestamp(yr, mon, 1)

def normnum(x):
    if x is None: return None
    s = str(x).strip().replace(".","").replace(",",".").replace("%","")
    try: return float(s)
    except: return None

with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[1]  # página 2 (1-based)
    tbls = page.extract_tables()
    if not tbls: raise SystemExit("No se detectaron tablas en p.2")
    df = pd.DataFrame(tbls[0])

# localizar la fila cabecera (busca "Mes" en col 0) y comprimir pares de columnas
hdr_i = 0
for i in range(min(5, len(df))):
    if isinstance(df.iloc[i,0], str) and "Mes" in df.iloc[i,0]:
        hdr_i = i; break

hdr = df.iloc[hdr_i].fillna("")
cols = []
i=0
while i < len(hdr):
    t1 = str(hdr[i]).strip()
    t2 = str(hdr[i+1]).strip() if i+1 < len(hdr) else ""
    cols.append((" ".join([t for t in (t1,t2) if t]) or f"col{i}").strip())
    i += 2 if t2 else 1

rows = []
for r in range(hdr_i+1, len(df)):
    raw = df.iloc[r].tolist()
    if all((x is None or str(x).strip()=="") for x in raw): continue
    compact = []
    i=0
    while i < len(raw):
        a = raw[i]
        b = raw[i+1] if i+1 < len(raw) else None
        pick = a if (a is not None and str(a).strip()!="") else b
        compact.append(pick)
        i += 2 if (b is not None and str(b).strip()!="") else 1
    while len(compact) < len(cols): compact.append(None)
    rows.append(compact[:len(cols)])

rd = pd.DataFrame(rows, columns=cols)

# columnas objetivo
month_col = None
for c in rd.columns:
    if rd[c].astype(str).str.contains(r"[a-zA-Z]{3}\s?-\s?\d{2,4}", regex=True).any():
        month_col = c; break
month_col = month_col or rd.columns[0]

def find_col(keys):
    for c in rd.columns:
        t = str(c).lower()
        if all(k in t for k in keys): return c
    return None

c_imp  = find_col(["importa"]) or find_col(["importación"]) or find_col(["importacion"])
c_tot  = find_col(["total"])
c_pct  = None
for c in rd.columns:
    if "%" in str(c): c_pct = c; break

# construir series
date = rd[month_col].map(parse_month)
out = {}

if c_imp:
    s = rd[[month_col, c_imp]].copy()
    s["_date"] = date; s["value"] = s[c_imp].map(normnum)
    out["mercado_importacion.csv"] = s[["_date","value"]].dropna().rename(columns={"_date":"date"})

if c_tot:
    s = rd[[month_col, c_tot]].copy()
    s["_date"] = date; s["value"] = s[c_tot].map(normnum)
    out["mercado_total.csv"] = s[["_date","value"]].dropna().rename(columns={"_date":"date"})

# % importados directo si aparece
if c_pct:
    s = rd[[month_col, c_pct]].copy()
    s["_date"] = date; 
    # valores con % en celdas
    s["value"] = s[c_pct].apply(lambda x: None if x is None else str(x).replace("%","").replace(".","").replace(",",".")).astype(float)
    out["mercado_pct_importados.csv"] = s[["_date","value"]].dropna().rename(columns={"_date":"date"})

# si no hay % importados, derivarlo = importacion/total*100
if "mercado_pct_importados.csv" not in out and {"mercado_importacion.csv","mercado_total.csv"} <= set(out.keys()):
    m = pd.merge(out["mercado_importacion.csv"], out["mercado_total.csv"], on="date", suffixes=("_imp","_tot"))
    m = m[m["value_tot"]!=0]
    m["value"] = (m["value_imp"]/m["value_tot"])*100
    out["mercado_pct_importados.csv"] = m[["date","value"]]

for name, s in out.items():
    s.sort_values("date").to_csv(OUT / name, index=False)
    print(f"OK -> {OUT / name} ({len(s)} filas)")
