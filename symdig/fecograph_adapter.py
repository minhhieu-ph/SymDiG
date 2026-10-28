"""Run the user's FeCoGraph re-implementation (github.com/minhhieu-ph/LineGraph) on OUR split.

Its train.run_once(args, run_id) obtains data from `prepare(args)`; we replace that function
inside the imported module with one that returns a GraphData built from our protocol-2 split
(same subsample, same labeled/val/test targets, same features as every other model), and build
the line graph with LineGraph's own `build_line_graph` on IP:PORT host keys (paper setting).
Nothing in the LineGraph repository is modified.
"""
import importlib
import os
import sys

import numpy as np

from .metrics import classification_metrics


def _import_linegraph(repo):
    repo = os.path.abspath(repo)
    if not os.path.exists(os.path.join(repo, "train.py")):
        raise FileNotFoundError(f"LineGraph repo not found at {repo} (git clone https://github.com/minhhieu-ph/LineGraph)")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    train = importlib.import_module("train")
    data_mod = importlib.import_module("src.data")
    return train, data_mod


def run_fecograph(cfg, data, split, seed, log=print, out_dir=None):
    if split.name != "p2":
        raise ValueError("FeCoGraph builds a full-batch line graph; run it on protocol p2 only")
    fc = cfg["fecograph"]
    train, data_mod = _import_linegraph(cfg["paths"]["linegraph_repo"])
    sub = np.sort(split.graph_train)
    pos = lambda ids: np.searchsorted(sub, np.sort(ids)).astype(np.int64)
    store = data.make_store(sub, cfg["data"].get("ports", True), seed=0)
    x = store.gather(sub)
    y = data.y[sub].astype(np.int64)
    d = data.dir
    sp = np.load(os.path.join(d, "src_ipport.npy"), mmap_mode="r")[sub]
    dp = np.load(os.path.join(d, "dst_ipport.npy"), mmap_mode="r")[sub]
    ei = data_mod.build_line_graph(np.asarray(sp), np.asarray(dp), max_clique=int(fc.get("max_clique", 100)),
                                   k_sample=int(fc.get("k_sample", 10)), seed=0)
    g = data_mod.GraphData(x=x, y=y, edge_index=ei, class_names=list(data.classes),
                           feature_names=data.feature_names(store.ports),
                           train_idx=pos(split.train), val_idx=pos(split.val), test_idx=pos(split.test))
    log(f"  [fecograph] line graph: {len(sub):,} flow-nodes, {ei.shape[1] // 2:,} undirected edges")
    argv = ["train.py", "--data", "unused.csv", "--task", data.task, "--seed", str(seed),
            "--epochs", str(fc.get("epochs", 2000)), "--eval-every", str(fc.get("eval_every", 10)),
            "--patience", str(fc.get("patience", 30)), "--out", out_dir or "results/fecograph_tmp",
            "--loss", fc.get("loss", "supcon"), "--lam", str(fc.get("lam", 0.5)), "--tau", str(fc.get("tau", 0.3))]
    old = sys.argv
    try:
        sys.argv = argv
        args = train.get_args()
    finally:
        sys.argv = old
    train.setup_perf(args)
    train.prepare = lambda a, verbose=True: g  # hand our split to the unmodified training loop
    os.makedirs(args.out, exist_ok=True)
    r = train.run_once(args, 0)
    # recompute our metrics from the per-class report for consistency with the other models
    rep = r["per_class"]
    per = {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"],
               "support": int(rep[c]["support"])} for c in data.classes if c in rep}
    present = [c for c in per if per[c]["support"] > 0]
    metrics = {"macro_f1": float(np.mean([per[c]["f1"] for c in present])),
               "macro_precision": float(np.mean([per[c]["precision"] for c in present])),
               "macro_recall": float(np.mean([per[c]["recall"] for c in present])),
               "accuracy": float(r["test"]["acc"]), "weighted_f1": float(rep.get("weighted avg", {}).get("f1-score", np.nan)),
               "per_class": per}
    return {"metrics": metrics, "hardware": {"peak_vram_gb": r.get("peak_vram_gib"), "train_sec": r.get("train_time_s")},
            "line_graph_edges": int(r.get("num_edges", 0)), "best_val": r.get("best_val"), "classes": data.classes}
