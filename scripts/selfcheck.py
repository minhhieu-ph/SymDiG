"""Quick environment + code check (1-3 minutes). Run this first on the GPU machine.

  python scripts/selfcheck.py            # uses the GPU if available
  python scripts/selfcheck.py --cpu

Checks: package versions; every model architecture x conv x sampler mode does a forward,
backward and predict on a tiny synthetic motif graph (bf16 autocast on GPU); XGBoost on GPU;
Optuna; the LineGraph (FeCoGraph) repository import.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ok = True
    print(f"python {sys.version.split()[0]}")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "yaml", "torch", "torch_geometric", "xgboost", "optuna", "pynvml",
                "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  {mod:16s} {getattr(m, '__version__', 'ok')}")
        except ImportError as ex:
            print(f"  {mod:16s} MISSING ({ex})")
            ok &= mod in ("optuna", "pynvml")  # optional
    if sys.version_info < (3, 9):
        print("Python >= 3.9 is required")
        return 1
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}" +
          (f" ({torch.cuda.get_device_name(0)}, bf16={torch.cuda.is_bf16_supported()})" if torch.cuda.is_available() else ""))

    from symdig.data import NetFlowData, get_split
    from symdig.jobs import load_config
    from symdig.planner import smoke_config
    from symdig.trainer import setup_perf, train_and_evaluate

    tmp = tempfile.mkdtemp(prefix="symdig_check_")
    try:
        cfg = smoke_config(load_config("configs/default.yaml"))
        cfg["paths"]["cache"] = tmp
        cfg["motif"]["n_nodes"] = 300
        cfg["runtime"] = {"threads": 2, "num_workers": 2}
        cfg["train"]["num_workers"] = 2
        setup_perf(2)
        data = NetFlowData(cfg, "motif_m8", log=lambda *x: None)
        split = get_split(cfg, data, "p1A")
        cases = [("hetero", False, False, "concat", "gine"), ("hetero", False, True, "concat", "gine"),
                 ("hetero", True, True, "concat", "gatv2"), ("dual", True, True, "concat", "gine"),
                 ("dual", True, True, "mha", "gatv2"), ("dir", True, True, "concat", "gine")]
        for arch, coupled, force, fusion, conv in cases:
            c = smoke_config(cfg)
            c["paths"]["cache"] = tmp
            c["runtime"] = cfg["runtime"]
            c["train"]["num_workers"] = 2
            c["model"].update(arch=arch, coupled=coupled, force=force, fusion=fusion, conv=conv, hidden=32)
            c["data"]["ports"] = False
            t0 = time.time()
            try:
                res, _, _ = train_and_evaluate(c, data, split, 0, log=lambda *x: None)
                f1 = res["metrics"]["macro_f1"]
                extra = f", covered-F1 {res['metrics_covered']['macro_f1']:.3f}, test coverage {res['test_seed_coverage']:.2f}" \
                    if "metrics_covered" in res and res["metrics_covered"] else ""
                print(f"  OK   {arch:6s} coupled={coupled!s:5s} force={force!s:5s} {fusion:6s} {conv:5s} "
                      f"macro-F1 {f1:.3f}{extra} ({time.time() - t0:.1f}s)")
            except Exception:
                ok = False
                print(f"  FAIL {arch} coupled={coupled} force={force} {fusion} {conv}")
                traceback.print_exc()
        # E-GraphSAGE on IP:PORT ids (motif data uses the same ids)
        try:
            c = smoke_config(cfg)
            c["paths"]["cache"] = tmp
            c["runtime"] = cfg["runtime"]
            c["train"]["num_workers"] = 0
            c["model"].update(arch="egsage", coupled=True, force=True, ego=False, edge_updates=False, hidden=32)
            d2 = NetFlowData(c, "motif_m8", node_mode="ipport", log=lambda *x: None)
            res, _, _ = train_and_evaluate(c, d2, get_split(c, d2, "p1A"), 0, log=lambda *x: None)
            print(f"  OK   egsage macro-F1 {res['metrics']['macro_f1']:.3f}")
        except Exception:
            ok = False
            print("  FAIL egsage")
            traceback.print_exc()
        # XGBoost on GPU
        try:
            import numpy as np
            from symdig.tabular import fit_predict
            X = np.random.rand(2000, 8).astype(np.float32)
            y = (X[:, 0] > 0.5).astype(np.int64)
            p, _, _, info = fit_predict("xgboost", {"n_estimators": 20}, X, y, X, y, X, 2, 0, 2, log=print)
            print(f"  OK   xgboost (acc {(p.argmax(1) == y).mean():.3f})")
        except Exception:
            ok = False
            traceback.print_exc()
        lg = cfg["paths"]["linegraph_repo"]
        print(f"  LineGraph repo for FeCoGraph: {'found' if os.path.exists(os.path.join(lg, 'train.py')) else 'NOT found at ' + lg}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELF-CHECK PASSED" if ok else "SELF-CHECK FAILED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
