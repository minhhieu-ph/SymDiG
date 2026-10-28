"""Loading and aggregating run outputs (shared by make_tables.py and make_figures.py)."""
import glob
import json
import os

import numpy as np

DS_ORDER = ["ton", "unsw", "ids2018", "bot"]
DS_SHORT = {"ton": "ToN", "unsw": "UNSW", "ids2018": "IDS2018", "bot": "BoT"}
MODEL_NAMES = {"symdig": "SymDiG (ours)", "multignn": "Multi-GNN", "multignn_fixed": "Multi-GNN + sampler fix",
               "dirgnn": "Dir-GNN", "egsage": "E-GraphSAGE", "xgboost": "XGBoost", "rf": "Random Forest",
               "mlp": "MLP", "fecograph": "FeCoGraph"}
RUNG_NAMES = {0: "(0) Multi-GNN pipeline", 1: "(1) + targets always sampled", 2: "(2) + coupled fwd/rev edges",
              3: "(3) + decoupled streams, concat", 4: "(4) + attentive fusion (SymDiG)"}


def load_runs(root):
    out = []
    for f in glob.glob(os.path.join(root, "runs", "*.json")):
        try:
            out.append(json.load(open(f)))
        except json.JSONDecodeError:
            continue
    return out


def ports_of(r):
    cfg = r.get("cfg") or {}
    return bool((cfg.get("data") or {}).get("ports", True))


def mstat(vals, scale=100.0, nd=2):
    v = np.array([x for x in vals if x is not None], float) * scale
    if len(v) == 0:
        return None, None, "--"
    if len(v) == 1:
        return v.mean(), 0.0, f"{v.mean():.{nd}f}"
    return v.mean(), v.std(ddof=1), f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def select(runs, **kw):
    out = []
    for r in runs:
        s = r["spec"]
        ok = True
        for k, v in kw.items():
            if k == "ports":
                ok &= ports_of(r) == v
            elif k == "preset":
                ok &= s.get("preset") == v
            else:
                ok &= s.get(k) == v
        if ok:
            out.append(r)
    return out


def macro(rs, key="metrics"):
    return [((r["result"] or {}).get(key) or {}).get("macro_f1") for r in rs]


def paired_test(a, b):
    """Paired differences over common seeds: mean diff and two-sided sign-flip permutation p-value."""
    da = {r["spec"]["seed"]: r["result"]["metrics"]["macro_f1"] for r in a if r["result"].get("metrics")}
    db = {r["spec"]["seed"]: r["result"]["metrics"]["macro_f1"] for r in b if r["result"].get("metrics")}
    common = sorted(set(da) & set(db))
    if len(common) < 2:
        return None, None, len(common)
    d = np.array([db[s] - da[s] for s in common]) * 100
    obs = abs(d.mean())
    n = len(d)
    signs = np.array([[1 if (i >> j) & 1 else -1 for j in range(n)] for i in range(2 ** n)])
    p = float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())
    return float(d.mean()), p, n
