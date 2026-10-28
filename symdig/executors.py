"""One function per job kind. Each returns a JSON-serialisable result dict."""
import copy
import json
import os
import time

import numpy as np

from .data import (FeatureStore, NetFlowData, Split, get_split, graph_for, line_graph_size, prepare,
                   subset_split, _strat)
from .jobs import deep_update, set_dotted
from .metrics import classification_metrics


# --------------------------------------------------------------------------- config resolution
def resolve(cfg, spec, results_root):
    """base config <- model preset / ladder rung <- named preset <- tuned params <- spec overrides."""
    c = copy.deepcopy(cfg)
    c["train"]["num_workers"] = int(c["runtime"].get("num_workers", 2))
    model = spec.get("model")
    node_mode = "ip"
    if model and model.startswith("ladder"):
        lad = {int(k): v for k, v in c["ladder"].items()}
        deep_update(c["model"], lad[int(model[6:])])
    elif model:
        m = c["models"][model]
        deep_update(c["model"], m.get("model", {}))
        node_mode = m.get("node_mode", "ip")
    if spec.get("preset"):
        deep_update(c, c[spec["preset"]])
    if spec.get("tuned"):
        f = os.path.join(results_root, "tuned", f"{spec['tuned']}.json")
        if not os.path.exists(f):
            raise FileNotFoundError(f"tuned parameters {f} missing (run the tune jobs first)")
        tuned = json.load(open(f))["best_params"]
        if c["models"][model]["kind"] == "tabular":
            c.setdefault("tabular_params", {}).update(tuned)
        else:
            for k, v in tuned.items():
                set_dotted(c, k, v)
    for k, v in (spec.get("overrides") or {}).items():
        set_dotted(c, k, v)
    c["_node_mode"] = node_mode
    return c


def _data(c, spec, task=None, dataset=None):
    return NetFlowData(c, dataset or spec["dataset"], task=task or spec.get("task", "multi"),
                       node_mode=c.get("_node_mode", "ip"))


def tabular_params(c, model):
    p = dict(c["tabular_defaults"][model])
    p.update(c.get("tabular_params", {}))
    return p


def _use_source_sampler(c, root, source_job):
    """Evaluate a trained checkpoint with the sampler settings it was trained with."""
    rec = json.load(open(os.path.join(root, "runs", f"{source_job}.json")))
    if rec.get("cfg") and rec["cfg"].get("sampler"):
        c["sampler"] = rec["cfg"]["sampler"]


# --------------------------------------------------------------------------- kinds
def run_prepare(cfg, spec, root, log):
    t0 = time.time()
    prepare(cfg, spec["dataset"], log=log)
    data = NetFlowData(cfg, spec["dataset"], log=log)
    out = {"n_edges": data.n_edges, "classes": data.classes, "class_counts": data.meta["class_counts"],
           "num_nodes": data.meta["num_nodes"], "sec": time.time() - t0}
    names = ["p1A", "p2"] + (["p1B"] if not spec["dataset"].startswith("motif") else [])
    for name in names:
        sp = get_split(cfg, data, name)
        out[f"split_{name}"] = sp.meta
    return out


def run_coverage(cfg, spec, root, log):
    from .sampler import coverage_stats
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, spec["split"])
    g = graph_for(data, split, "train")
    cfg_s = dict(c["sampler"], fanout=spec["fanout"])
    out = {"fanout": spec["fanout"], "graph": g.stats()}
    for name, mode in [("multignn", {"coupled": False, "force": False}),
                       ("asym_forced", {"coupled": False, "force": True}),
                       ("coupled_forced", {"coupled": True, "force": True})]:
        bs = 8192 if spec["fanout"][0] >= 100 else int(c["train"]["batch_size"])
        out[name] = coverage_stats(g, split.train, mode, cfg_s, bs, int(spec["batches"]), seed=0)
        log(f"  {name}: seed coverage {out[name]['seed_coverage']:.3f}, twin coverage {out[name]['twin_coverage']:.3f}")
    return out


def run_survival(cfg, spec, root, log):
    from .motifs import survival
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, "p1A")
    g = graph_for(data, split, "train")
    r = survival(data, g, spec["fanout"])
    r["multiplicity"] = data.meta.get("multiplicity")
    log(f"  survival: {r}")
    return r


def run_gnn(cfg, spec, root, log, return_model=False):
    from .trainer import train_and_evaluate
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, spec["split"])
    ck = os.path.join(root, "ckpt", f"{spec['id']}.pt") if spec.get("id") else None
    res, model, store = train_and_evaluate(c, data, split, int(spec["seed"]), log=log, ckpt_path=ck)
    res["split_meta"] = split.meta
    res["model_cfg"] = c["model"]
    if return_model:
        return res, model, store, c, data, split
    return res


def run_tabular(cfg, spec, root, log):
    from .tabular import run_tabular as rt
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, spec["split"])
    mp = os.path.join(root, "models", f"{spec['id']}.json") if spec.get("id") else None
    res, _, _ = rt(c, data, split, spec["model"], tabular_params(c, spec["model"]), int(spec["seed"]), log, mp)
    res["params"] = tabular_params(c, spec["model"])
    res["split_meta"] = split.meta
    return res


def run_fecograph(cfg, spec, root, log):
    from .fecograph_adapter import run_fecograph as rf
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, spec["split"])
    return rf(c, data, split, int(spec["seed"]), log, out_dir=os.path.join(root, "fecograph_raw", spec["id"]))


def run_tune(cfg, spec, root, log):
    from .tuning import tune
    return tune(cfg, spec, root, log)


def run_transfer(cfg, spec, root, log):
    """Binary detector trained on spec['source'] (split p1A) evaluated on spec['dataset']'s test split."""
    c = resolve(cfg, spec, root)
    kind = c["models"][spec["model"]]["kind"]
    src = NetFlowData(c, spec["source"], task="binary", node_mode=c["_node_mode"])
    tgt = NetFlowData(c, spec["dataset"], task="binary", node_mode=c["_node_mode"])
    ports = c["data"].get("ports", True)
    if src.feature_names(ports) != tgt.feature_names(ports):
        raise ValueError("source and target datasets do not share the same feature columns")
    sp_t = get_split(c, tgt, "p1A")
    te = np.sort(sp_t.test)
    if kind == "gnn":
        from .trainer import load_checkpoint, predict
        _use_source_sampler(c, root, spec["source_job"])
        model, ck = load_checkpoint(os.path.join(root, "ckpt", f"{spec['source_job']}.pt"))
        st = ck["store"]
        store = FeatureStore(tgt.X, tgt.sport, tgt.dport, st["ports"], st["mean"], st["std"])
        g = graph_for(tgt, sp_t, "test")
        probs, _, _ = predict(model, g, store, tgt.y, te, {"coupled": bool(ck["model_cfg"]["coupled"]), "force": True},
                              c, int(spec["seed"]))
    else:
        from .tabular import fit_predict
        sp_s = get_split(c, src, "p1A")
        store = src.make_store(sp_s.graph_train, ports, seed=0)
        Xs, ys = store.gather(sp_s.train), src.y[sp_s.train]
        va = np.sort(sp_s.val)
        tstore = FeatureStore(tgt.X, tgt.sport, tgt.dport, store.ports, store.mean, store.std)
        probs, _, _, _ = fit_predict(spec["model"], tabular_params(c, spec["model"]), Xs, ys, store.gather(va),
                                     src.y[va], tstore.gather(te), 2, int(spec["seed"]),
                                     int(c["runtime"].get("threads", 4)), log)
    m = classification_metrics(tgt.y[te], probs, tgt.classes)
    log(f"  transfer {spec['source']} -> {spec['dataset']}: macro-F1 {m['macro_f1']:.4f}")
    return {"metrics": m, "source": spec["source"], "target": spec["dataset"]}


def run_sizes(cfg, spec, root, log):
    """E6: graph sizes on the same flows for the three graph constructions."""
    data = NetFlowData(cfg, spec["dataset"])
    d = data.dir
    ld = lambda n: np.asarray(np.load(os.path.join(d, f"{n}.npy"), mmap_mode="r")).astype(np.int64)
    sip, dip, spp, dpp = ld("src_ip"), ld("dst_ip"), ld("src_ipport"), ld("dst_ipport")
    rng = np.random.default_rng(0)
    out = {}
    for f in spec["fracs"]:
        idx = _strat(data.y_multi, np.arange(data.n_edges), float(f), rng, min_keep=0) if f < 1 else np.arange(data.n_edges)
        s1, d1, s2, d2 = sip[idx], dip[idx], spp[idx], dpp[idx]
        key = s1 * (int(dip.max()) + 1) + d1
        out[str(f)] = {"flows": int(len(idx)),
                       "ip_nodes": int(len(np.unique(np.concatenate([s1, d1])))),
                       "ipport_nodes": int(len(np.unique(np.concatenate([s2, d2])))),
                       "ip_pairs": int(len(np.unique(key))),
                       "line_graph_edges_ipport": line_graph_size(s2, d2),
                       "line_graph_edges_ip": line_graph_size(s1, d1)}
        log(f"  {f}: {out[str(f)]}")
    return out


def run_deploy(cfg, spec, root, log):
    """E6: training VRAM / step time and inference throughput / latency (accuracy irrelevant)."""
    import torch
    from .loader import SubgraphBatches, make_loader, to_device
    from .trainer import autocast, get_device, train_and_evaluate
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, "p1A")
    if float(spec.get("frac", 1.0)) < 1.0:
        split = subset_split(split, float(spec["frac"]), data.y_multi, seed=0)
    dev = get_device()
    cap = spec.get("mem_cap_gb")
    out = {"frac": spec.get("frac", 1.0), "mem_cap_gb": cap, "model": spec["model"]}
    if cap and dev.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) / total))
    try:
        prof, model, store = train_and_evaluate(c, data, split, 0, log=log, profile_steps=int(spec["steps"]))
        out["train"] = prof["hardware"]
        g = graph_for(data, split, "test")
        mode = {"coupled": bool(c["model"]["coupled"]), "force": True}
        lat, n_flows = [], 0
        for bs in (1024, int(c["train"]["eval_batch_size"])):
            ds = SubgraphBatches(g, store, data.y, np.sort(split.test)[: bs * 20], bs, mode, c["sampler"],
                                 shuffle=False, seed=0)
            times = []
            model.eval()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t_all = time.time()
            for b in make_loader(ds, c["train"]["num_workers"]):
                t1 = time.time()
                b = to_device(b, dev)
                with torch.no_grad(), autocast(c["train"]["amp"], dev):
                    model(b)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t1)
            wall = time.time() - t_all
            out[f"infer_bs{bs}"] = {"gpu_ms_p50": 1000 * float(np.median(times)), "gpu_ms_p95": 1000 * float(np.percentile(times, 95)),
                                    "flows_per_sec_end_to_end": bs * len(times) / max(wall, 1e-9),
                                    "flows_per_sec_gpu": bs / max(float(np.median(times)), 1e-9),
                                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None}
        out["status"] = "ok"
    except RuntimeError as ex:  # CUDA OOM under the memory cap is a result, not a failure
        if "out of memory" not in str(ex).lower():
            raise
        out["status"] = "OOM"
        log(f"  OOM: {ex}")
    finally:
        if cap and dev.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(1.0)
            torch.cuda.empty_cache()
    out["graph"] = graph_for(data, split, "train").stats()
    return out


def run_evasion(cfg, spec, root, log):
    """E7. Per attacked class F1 under injection; non-attacked test targets are subsampled
    (rate q) and their predicted-positive counts re-weighted by 1/q when computing precision."""
    from .evasion import inject
    from .trainer import load_checkpoint, predict
    c = resolve(cfg, spec, root)
    data = _data(c, spec)
    split = get_split(c, data, "p1A")
    _use_source_sampler(c, root, spec["source_job"])
    model, ck = load_checkpoint(os.path.join(root, "ckpt", f"{spec['source_job']}.pt"))
    st = ck["store"]
    store = FeatureStore(data.X, data.sport, data.dport, st["ports"], st["mean"], st["std"])
    attacked = [x for x in spec["classes"] if x in data.classes]
    cid = np.array([data.classes.index(x) for x in attacked])
    te = np.sort(split.test)
    att = te[np.isin(data.y[te], cid)]
    other = te[~np.isin(data.y[te], cid)]
    rng = np.random.default_rng(int(spec["seed"]))
    q = min(1.0, 500_000 / max(len(other), 1))
    oth = np.sort(rng.choice(other, int(len(other) * q), replace=False)) if q < 1 else other
    targets = np.sort(np.concatenate([att, oth]))
    weight = np.where(np.isin(targets, att), 1.0, 1.0 / q)
    mode = {"coupled": bool(ck["model_cfg"]["coupled"]), "force": True}
    out = {"attacked": attacked, "q_other": q, "n_attacked": int(len(att)), "results": {}}

    def score(g, sto):
        probs, _, _ = predict(model, g, sto, data.y, targets, mode, c, int(spec["seed"]))
        pred, y = probs.argmax(1), data.y[targets]
        r = {}
        for name, k in zip(attacked, cid):
            tp = float(((pred == k) & (y == k)).sum())
            fp = float((weight * ((pred == k) & (y != k))).sum())
            fn = float(((pred != k) & (y == k)).sum())
            r[name] = {"f1": 2 * tp / max(2 * tp + fp + fn, 1e-9), "recall": tp / max(tp + fn, 1e-9),
                       "precision": tp / max(tp + fp, 1e-9)}
        r["mean_f1"] = float(np.mean([r[n]["f1"] for n in attacked]))
        return r

    base_g = graph_for(data, split, "test")
    clean = score(base_g, store)
    for strat in spec["strategies"]:
        out["results"][strat] = {}
        for k in spec["k"]:
            if int(k) == 0:
                out["results"][strat]["0"] = clean
                continue
            g, sto = inject(data, store, np.asarray(split.graph_test), att, strat, int(k), seed=int(spec["seed"]))
            out["results"][strat][str(k)] = score(g, sto)
            log(f"  {strat} K={k}: mean attacked-class F1 {out['results'][strat][str(k)]['mean_f1']:.4f}")
    return out


EXECUTORS = {"prepare": run_prepare, "coverage": run_coverage, "survival": run_survival, "gnn": run_gnn,
             "tabular": run_tabular, "fecograph": run_fecograph, "tune": run_tune, "transfer": run_transfer,
             "sizes": run_sizes, "deploy": run_deploy, "evasion": run_evasion}


def resolved_cfg_for_record(cfg, spec, root):
    try:
        if spec["kind"] in ("gnn", "tabular", "fecograph", "deploy", "transfer", "coverage"):
            c = resolve(cfg, spec, root)
            return {k: c[k] for k in ("model", "sampler", "train", "data") if k in c} | \
                   ({"tabular_params": tabular_params(c, spec["model"])} if spec.get("kind") == "tabular" else {})
    except Exception:
        pass
    return None
