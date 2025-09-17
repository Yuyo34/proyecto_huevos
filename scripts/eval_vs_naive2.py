import pandas as pd, numpy as np, os

TARGET = "data/precio_huevo_mensual_real.csv"
PRED_FILES = ["out/rocv_h1_pred.csv","out/rocv_h2_pred.csv"]

PREF_PRED_NAMES = ["yhat","forecast","pred","prediction","y_pred","fcst","hat","value_pred"]

def load_target(path):
    df = pd.read_csv(path)
    # detectar columnas (date + value)
    d = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    v = next((c for c in df.columns if c != d), None)
    if v is None: 
        raise SystemExit(f"No encuentro columna de valores en {path}. Cols={list(df.columns)}")
    s = pd.Series(pd.to_numeric(df[v], errors="coerce").values,
                  index=pd.to_datetime(df[d], errors="coerce")).dropna()
    # index mensual, inicio de mes
    s.index = s.index.to_period("M").to_timestamp(how="start")
    # completar huecos intrames (si los hubiera)
    s = s.asfreq("MS")
    return s

def detect_date_col(df):
    # preferidos por nombre
    for c in df.columns:
        if c.lower() in ["date","ds","fecha"]:
            return c
    # por parsabilidad a datetime
    best, best_rate = None, 0.0
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        rate = s.notna().mean()
        if rate > best_rate:
            best, best_rate = c, rate
    return best

def detect_pred_col(df, date_col):
    # 1) por nombre preferido
    for name in PREF_PRED_NAMES:
        for c in df.columns:
            if c.lower() == name:
                return c
    # 2) primera numérica distinta de fecha
    num = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if num:
        return num[-1]  # última numérica (suele ser la predicción en nuestros CSV)
    # 3) castear y buscar numéricas
    cand = []
    for c in df.columns:
        if c == date_col: 
            continue
        try:
            pd.to_numeric(df[c], errors="coerce")
            cand.append(c)
        except Exception:
            pass
    return cand[-1] if cand else None

def eval_file(pred_path, y):
    base = os.path.basename(pred_path)
    if not os.path.exists(pred_path):
        return {"file": base, "rows": 0, "pairs": 0, "mae_model": None, "mae_naive": None, "rMAE": None}

    df = pd.read_csv(pred_path)
    if df.empty:
        return {"file": base, "rows": 0, "pairs": 0, "mae_model": None, "mae_naive": None, "rMAE": None}

    dcol = detect_date_col(df) or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).copy()
    # normaliza a mensual (inicio de mes)
    df[dcol] = df[dcol].dt.to_period("M").dt.to_timestamp(how="start")

    pcol = detect_pred_col(df, dcol)
    if pcol is None:
        return {"file": base, "rows": int(df.shape[0]), "pairs": 0, "mae_model": None, "mae_naive": None, "rMAE": None}

    # verdad-terreno desde el TARGET alineado por fecha
    y_true = y.reindex(df[dcol].values)
    y_naiv = y.shift(12).reindex(df[dcol].values)  # y_{t-12}
    y_hat  = pd.to_numeric(df[pcol], errors="coerce").values

    mask = (~y_true.isna()) & (~y_naiv.isna()) & (~np.isnan(y_hat))
    if mask.sum() == 0:
        return {"file": base, "rows": int(df.shape[0]), "pairs": 0, "mae_model": None, "mae_naive": None, "rMAE": None}

    yt = y_true[mask].values
    yh = y_hat[mask]
    yn = y_naiv[mask].values

    mae_model = float(np.mean(np.abs(yt - yh)))
    mae_naive = float(np.mean(np.abs(yt - yn)))
    rmae = (mae_model/mae_naive) if mae_naive > 0 else np.nan

    return {
        "file": base,
        "rows": int(df.shape[0]),
        "pairs": int(mask.sum()),
        "date_min": str(df[dcol].min().date()),
        "date_max": str(df[dcol].max().date()),
        "pred_col": pcol,
        "mae_model": round(mae_model,3),
        "mae_naive": round(mae_naive,3),
        "rMAE": round(rmae,3) if rmae==rmae else None
    }

def main():
    y = load_target(TARGET)
    rows = [eval_file(p, y) for p in PRED_FILES]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("out/rocv_eval_vs_naive.csv", index=False)
    print("\nOK -> out/rocv_eval_vs_naive.csv")

if __name__ == "__main__":
    main()
