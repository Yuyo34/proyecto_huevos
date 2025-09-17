import pandas as pd, numpy as np, os

TARGET = "data/precio_huevo_mensual_real.csv"
PRED_FILES = ["out/rocv_h1_pred.csv","out/rocv_h2_pred.csv"]

def load_target(path):
    df = pd.read_csv(path)
    d = next((c for c in df.columns if c.lower() in ["date","ds","fecha"]), df.columns[0])
    v = next((c for c in df.columns if c != d), None)
    s = pd.Series(pd.to_numeric(df[v], errors="coerce").values,
                  index=pd.to_datetime(df[d], errors="coerce")).dropna()
    # A mensual (inicio de mes) sin usar "MS" en PeriodIndex:
    s.index = s.index.to_period("M").to_timestamp(how="start")
    return s

def pick(df, kind):
    opts = dict(
        date=["date","ds","fecha"],
        truth=["y_true","actual","y","target","obs","real","value_true","value","actuals"],
        pred=["yhat","forecast","pred","prediction","y_pred","fcst","hat","value_pred"]
    )[kind]
    for c in df.columns:
        if c.lower() in opts: return c
    return None

def eval_file(pred_path, y):
    if not os.path.exists(pred_path):
        return {"file": os.path.basename(pred_path), "rows": 0,
                "mae_model": None, "mae_naive": None, "rMAE": None, "pairs": 0}

    df = pd.read_csv(pred_path)
    d  = pick(df,"date")  or df.columns[0]
    yt = pick(df,"truth")
    yh = pick(df,"pred")
    if yt is None or yh is None:
        return {"file": os.path.basename(pred_path), "rows": int(df.shape[0]),
                "mae_model": None, "mae_naive": None, "rMAE": None, "pairs": 0}

    # Normaliza fechas del pred a mensual (inicio de mes)
    df[d] = pd.to_datetime(df[d], errors="coerce")
    df[d] = df[d].dt.to_period("M").dt.to_timestamp(how="start")
    df = df.dropna(subset=[d, yt, yh])

    # Naïve: y_{t-12} (mismo período del año anterior)
    df["y_naive"] = df[d].apply(lambda ts: y.get(ts - pd.offsets.DateOffset(years=1), np.nan))

    sub = df.loc[df["y_naive"].notna(), [yt, yh, "y_naive"]].astype(float)
    if len(sub) == 0:
        return {"file": os.path.basename(pred_path), "rows": int(df.shape[0]),
                "mae_model": None, "mae_naive": None, "rMAE": None, "pairs": 0}

    mae_model = float(np.mean(np.abs(sub[yt] - sub[yh])))
    mae_naive = float(np.mean(np.abs(sub[yt] - sub["y_naive"])))
    rmae = (mae_model/mae_naive) if mae_naive > 0 else np.nan

    return {"file": os.path.basename(pred_path),
            "rows": int(df.shape[0]),
            "pairs": int(len(sub)),
            "mae_model": round(mae_model,3),
            "mae_naive": round(mae_naive,3),
            "rMAE": round(rmae,3) if rmae==rmae else None}

def main():
    y = load_target(TARGET)
    rows = [eval_file(p, y) for p in PRED_FILES]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("out/rocv_eval_vs_naive.csv", index=False)
    print("\nOK -> out/rocv_eval_vs_naive.csv")

if __name__ == "__main__":
    main()
