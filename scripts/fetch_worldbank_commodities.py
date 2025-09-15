import re, io, urllib.request, pandas as pd, numpy as np
from pathlib import Path

WB_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
FALLBACK = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"

def find_monthly_xls():
    try:
        with urllib.request.urlopen(WB_PAGE) as r:
            html = r.read().decode("utf-8", errors="ignore")
        m = re.search(r'href="([^"]+Monthly[^"]+\.xls[x]?)"', html, flags=re.I)
        if m: return m.group(1)
    except Exception:
        pass
    return FALLBACK

def load_monthly_table(b):
    xls = pd.ExcelFile(io.BytesIO(b))
    sh = next((s for s in xls.sheet_names if "month" in s.lower()), xls.sheet_names[0])
    return xls.parse(sh, header=0)

def pick_series(df, key):
    # Busca la fila cuyo nombre contenga la palabra clave
    key = key.lower()
    row = df.loc[df.apply(lambda r: any(key in str(r[c]).lower() for c in df.columns[:3]), axis=1)]
    if row.empty: return pd.Series(dtype=float)
    row = row.iloc[0]
    # Columnas fecha: pandas detecta fechas si existen; si no, intenta parsear nombres
    s = {}
    for c in df.columns:
        try:
            ts = pd.to_datetime(c, errors="raise").to_period("M").to_timestamp("MS")
            val = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(val): s[ts] = float(val)
        except Exception:
            continue
    return pd.Series(s).sort_index()

def to_dlog_csv(s, out_path):
    s = s.dropna()
    s = s[s > 0]
    dlog = np.log(s).diff().dropna()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dlog.index, "value": dlog.values}).to_csv(out_path, index=False)

def main():
    url = find_monthly_xls()
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dfm = load_monthly_table(data)
    for key, out in [("maize","data/corn_dlog.csv"),
                     ("soybean","data/soy_dlog.csv"),
                     ("brent","data/brent_dlog.csv")]:
        s = pick_series(dfm, key)
        if not s.empty:
            to_dlog_csv(s, out)
            print(f"OK -> {out} [{s.index.min().date()}..{s.index.max().date()}]")

if __name__ == "__main__":
    main()
