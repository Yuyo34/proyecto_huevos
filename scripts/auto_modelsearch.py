import re, subprocess, shlex, csv, itertools, sys, shutil
from pathlib import Path

# === Config base ===
TARGET = Path("data/precio_huevo_mensual.csv")
BASE_EXOG = {
    "--usdclp": Path("data/usdclp_dlog.csv"),
    "--ipc":    Path("data/imacec_yoy.csv"),
}
OPT_EXOG = {
    "--corn":   Path("data/corn_dlog.csv"),
    "--soy":    Path("data/soy_dlog.csv"),
    "--diesel": Path("data/brent_dlog.csv"),
}
LAGS = [1]  # puedes aÃ±adir 2 si quieres explorar mÃ¡s
BT_INIT = 12
H = 2
SEASONALITY = ["add", "mult"]  # aditiva vs multiplicativa
NO_BOOST = True

PYMOD = ["py", "-m", "upgrade_precision.pipeline.pipeline_monthly_exog"]

def available_opt_exog():
    return {k:v for k,v in OPT_EXOG.items() if v.exists()}

def powerset(keys):
    ks = list(keys)
    return itertools.chain.from_iterable(itertools.combinations(ks, r) for r in range(len(ks)+1))

def run_cfg(exogs, season, lag):
    fname = f"auto_fcast_{season}_lag{lag}_" + ("none" if not exogs else "_".join(k.strip("-") for k in exogs)) + ".csv"
    fname = f"auto_fcast_{season}_lag{lag}_" + ("none" if not exogs else "_".join(k.strip("-") for k in exogs)) + ".csv"
    out = Path("out") / fname
    args = PYMOD + [
        "--target", str(TARGET),
        "--bt_init", str(BT_INIT),
        "--seasonality", "12",
        "--h", str(H),
        "--lags", str(lag),
        "--out", str(out),
    ]
    if season == "mult":
        args.append("--multiplicative")
    if NO_BOOST:
        args.append("--no_boost")

    # exÃ³genas base (si existen)
    for flag, path in BASE_EXOG.items():
        if path.exists():
            args += [flag, str(path)]
    # exÃ³genas opcionales (subset actual)
    for flag in exogs:
        args += [flag, str(OPT_EXOG[flag])]

    print("\n[RUN]", " ".join(shlex.quote(a) for a in args))
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(p.stdout)
    if p.returncode != 0:
        print("[ERR]", p.stderr.strip())
        return None

    # parsea "[MODEL] elegido: X (MASE=0.1234)"
    chosen = None
    mase = None
    last_m = None
    for last_m in re.finditer(r"\[MODEL\]\s+elegido:\s*([a-zA-Z0-9_]+)\s*\(MASE=([0-9.]+)\)", p.stdout):
        pass
    if last_m:
        chosen = last_m.group(1)
        mase = float(last_m.group(2))
    else:
        # fallback: Ãºltimo nÃºmero que aparezca como MASE en logs
        last = None
        for mm in re.finditer(r"MASE[=:\s]*([0-9.]+)", p.stdout):
            last = mm.group(1)
        if last:
            mase = float(last)

    return {
        "exogs": ",".join(k.strip("-") for k in exogs) if exogs else "(ninguna extra)",
        "seasonality": season,
        "lags": lag,
        "mase": mase if mase is not None else float("inf"),
        "model": chosen or "(desconocido)",
        "out_path": str(out),
    }

def main():
    if not TARGET.exists():
        print(f"[FATAL] no existe {TARGET}.")
        sys.exit(1)

    avail_opt = available_opt_exog()
    print("[INFO] OPT exog disponibles:", ", ".join(k for k in avail_opt.keys()) or "(ninguna)")

    results = []
    for season in SEASONALITY:
        for lag in LAGS:
            for subset in powerset(avail_opt.keys()):
                res = run_cfg(subset, season, lag)
                if res:
                    results.append(res)

    if not results:
        print("[FATAL] No hubo corridas vÃ¡lidas.")
        sys.exit(2)

    # ordenar por MASE ascendente
    results.sort(key=lambda r: r["mase"])
    best = results[0]

    # guardar resumen
    out_csv = Path("out/gridsearch_summary.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exogs","seasonality","lags","mase","model","out_path"])
        w.writeheader()
        w.writerows(results)

    # copiar el mejor a un alias estable
    best_alias = Path("out/forecast_next2m_best.csv")
    try:
        shutil.copyfile(best["out_path"], best_alias)
    except Exception:
        pass

    print("\n=== TOP 5 (por MASE) ===")
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. MASE={r['mase']:.4f} | model={r['model']} | season={r['seasonality']} | lags={r['lags']} | exogs={r['exogs']}")
    print(f"\n[OK] Mejor config -> MASE={best['mase']:.4f}, model={best['model']}, season={best['seasonality']}, lags={best['lags']}, exogs={best['exogs']}")
    print(f"[OK] Resumen: {out_csv}")
    print(f"[OK] PronÃ³stico (best): {best['out_path']}  (copia: {best_alias})")

if __name__ == "__main__":
    main()



