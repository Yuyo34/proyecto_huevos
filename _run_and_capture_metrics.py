import argparse, subprocess, sys, re, ast, json, os
from pathlib import Path
from datetime import datetime
import pandas as pd

"""
Este wrapper:
1) Ejecuta el pipeline upgrade_precision.
2) Captura stdout/stderr en out/run.log.
3) Extrae la línea [BT] ... métricas (ORIG): {...} y escribe out/metrics.json.
"""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--seasonality", type=int, default=12)
    ap.add_argument("--bt_init", type=int, default=12)
    ap.add_argument("--lags", nargs="+", type=int, default=[1])
    # exógenas (opcionales)
    ap.add_argument("--usdclp")
    ap.add_argument("--ipc")
    ap.add_argument("--diesel")
    ap.add_argument("--corn")
    ap.add_argument("--soy")
    ap.add_argument("--multiplicative", action="store_true")
    return ap.parse_args()

def build_cmd(a):
    cmd = [sys.executable, "-m", "upgrade_precision.pipeline.pipeline_monthly_exog",
           "--target", a.target, "--h", str(a.h), "--seasonality", str(a.seasonality),
           "--bt_init", str(a.bt_init), "--out", a.out]
    if a.lags: cmd += ["--lags", *map(str, a.lags)]
    for name in ["usdclp","ipc","diesel","corn","soy"]:
        val = getattr(a, name)
        if val: cmd += [f"--{name}", val]
    if a.multiplicative:
        cmd += ["--multiplicative"]
    return cmd

def extract_metrics(text):
    # Busca el dict después de: "[BT] ... métricas (ORIG): { ... }"
    m = re.search(r"\[BT\].*métricas \(ORIG\):\s*(\{.*\})", text, flags=re.DOTALL)
    if not m:
        return None
    s = m.group(1)
    # Quitar np.float64(…) -> …
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    try:
        d = ast.literal_eval(s)
        # convertir a float nativo
        d = {k: float(v) for k, v in d.items()}
        return d
    except Exception:
        return None

def main():
    a = parse_args()
    out_dir = Path(a.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(a)
    run = subprocess.run(cmd, capture_output=True, text=True)
    log = (run.stdout or "") + "\n--- STDERR ---\n" + (run.stderr or "")
    (out_dir / "run.log").write_text(log, encoding="utf-8")

    metrics = extract_metrics(log) or {}
    # Metadatos útiles
    sha = os.getenv("GITHUB_SHA") or ""
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_sha": sha,
        "params": {
            "bt_init": a.bt_init, "seasonality": a.seasonality, "h": a.h, "lags": a.lags,
            "multiplicative": a.multiplicative
        },
        "files": {"target": a.target, "out": a.out}
    }
    # Rango del target
    try:
        df = pd.read_csv(a.target, parse_dates=["date"])
        df = df.dropna(subset=["date","value"]).sort_values("date")
        meta["target_range"] = {
            "rows": int(len(df)),
            "min_date": df["date"].min().date().isoformat(),
            "max_date": df["date"].max().date().isoformat()
        }
    except Exception:
        pass

    payload = {"metrics": metrics, "meta": meta}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Mostrar resumen en stdout del workflow
    print("== METRICS ==")
    print(json.dumps(payload, indent=2))
    # Propagar exit code del pipeline (pero ya tenemos log/metrics aunque falle)
    sys.exit(run.returncode)
if __name__ == "__main__":
    main()
