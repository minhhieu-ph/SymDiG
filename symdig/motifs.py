"""E3: synthetic directed multigraphs with planted motifs (structure-only labels).

Edge features are i.i.d. Gaussian noise drawn identically for every class, so an edge's
label can only be recovered from the graph structure around it. Every logical edge is
replicated m ~ 1 + Poisson(M - 1) times (all copies share the label), so the
multiplicity M controls density without making multiplicity itself informative.

Classes (priority when an edge qualifies for several):
  3 cycle          the edge lies on a directed cycle of length 3 or 4
  4 scatter-gather u -> v -> w with >= K_SG distinct 2-paths from u to w (edges on either hop)
  1 fan-in         destination has >= K_FAN distinct in-neighbours
  2 fan-out        source has >= K_FAN distinct out-neighbours
  0 none
The cache it writes has the same layout as a NetFlow dataset, so every experiment runs on it.
"""
import json
import os

import numpy as np
import scipy.sparse as sp

from .data import CACHE_VERSION, _atomic_json, cache_dir

CLASSES = ["None", "FanIn", "FanOut", "Cycle", "ScatterGather"]
K_FAN, K_SG = 8, 4


def parse_key(key):
    # motif_m16  -> M = 16
    return int(key.split("_m")[1])


def generate(M, n_nodes=4000, bg_deg=2.0, n_fan=40, n_cyc=400, n_sg=120, d=8, seed=0):
    rng = np.random.default_rng(seed)
    E = set()

    def add(u, v):
        if u != v:
            E.add((int(u), int(v)))

    m_bg = int(bg_deg * n_nodes)
    for u, v in zip(rng.integers(0, n_nodes, m_bg), rng.integers(0, n_nodes, m_bg)):
        add(u, v)
    planted_cycles = []
    for _ in range(n_fan):
        h = rng.integers(n_nodes)
        for s in rng.choice(n_nodes, rng.integers(K_FAN, 2 * K_FAN), replace=False):
            add(s, h)
        h = rng.integers(n_nodes)
        for t in rng.choice(n_nodes, rng.integers(K_FAN, 2 * K_FAN), replace=False):
            add(h, t)
    for _ in range(n_cyc):
        L = rng.integers(3, 5)
        c = rng.choice(n_nodes, L, replace=False)
        for a, b in zip(c, np.roll(c, -1)):
            add(a, b)
        planted_cycles.append([int(x) for x in c])
    for _ in range(n_sg):
        u, w = rng.choice(n_nodes, 2, replace=False)
        for v in rng.choice(n_nodes, rng.integers(K_SG, K_SG + 3), replace=False):
            if v not in (u, w):
                add(u, v)
                add(v, w)
    edges = np.array(sorted(E), np.int64)
    u, v = edges[:, 0], edges[:, 1]
    A = sp.csr_matrix((np.ones(len(u)), (u, v)), shape=(n_nodes, n_nodes))
    A.data[:] = 1
    A2 = A @ A
    A3 = A2 @ A
    lab = np.zeros(len(u), np.int64)
    indeg = np.asarray(A.sum(0)).ravel()
    outdeg = np.asarray(A.sum(1)).ravel()
    lab[outdeg[u] >= K_FAN] = 2
    lab[indeg[v] >= K_FAN] = 1
    S = (A2 >= K_SG).astype(np.float64)
    S.setdiag(0)
    S.eliminate_zeros()
    first = np.asarray((S @ A.T)[u, v]).ravel() > 0     # exists w: S[u,w] and A[v,w]
    second = np.asarray((A.T @ S)[u, v]).ravel() > 0    # exists x: A[x,u] and S[x,v]
    lab[first | second] = 4
    on_cycle = (np.asarray(A2[v, u]).ravel() > 0) | (np.asarray(A3[v, u]).ravel() > 0)
    lab[on_cycle] = 3
    m = 1 + rng.poisson(max(M - 1, 0), len(u))
    src, dst, y = np.repeat(u, m), np.repeat(v, m), np.repeat(lab, m)
    perm = rng.permutation(len(src))
    src, dst, y = src[perm], dst[perm], y[perm]
    X = rng.standard_normal((len(src), d)).astype(np.float32)
    return {"X": X, "y": y, "src": src, "dst": dst, "n_nodes": n_nodes, "cycles": planted_cycles,
            "logical_labels": np.bincount(lab, minlength=len(CLASSES)).tolist()}


def prepare_motif(cfg, key, log=print):
    out = cache_dir(cfg, key)
    if os.path.exists(os.path.join(out, "meta.json")):
        return out
    os.makedirs(out, exist_ok=True)
    M = parse_key(key)
    mc = cfg.get("motif", {})
    g = generate(M, n_nodes=mc.get("n_nodes", 4000), seed=mc.get("seed", 0))
    sv = lambda n, a: np.save(os.path.join(out, f"{n}.npy"), a)
    sv("X", g["X"])
    sv("y", g["y"].astype(np.int16))
    sv("ybin", (g["y"] > 0).astype(np.int8))
    for mode in ("ip", "ipport"):
        sv(f"src_{mode}", g["src"].astype(np.int32))
        sv(f"dst_{mode}", g["dst"].astype(np.int32))
    with open(os.path.join(out, "cycles.json"), "w") as fh:
        json.dump(g["cycles"], fh)
    counts = np.bincount(g["y"], minlength=len(CLASSES))
    meta = {"version": CACHE_VERSION, "dataset": key, "name": f"Motif (M={M})", "n_edges": int(len(g["y"])),
            "classes": CLASSES, "class_counts": counts.tolist(), "logical_class_counts": g["logical_labels"],
            "num_nodes": {"ip": g["n_nodes"], "ipport": g["n_nodes"]},
            "feature_names": [f"noise{i}" for i in range(g["X"].shape[1])], "has_ports": False, "multiplicity": M}
    _atomic_json(os.path.join(out, "meta.json"), meta)
    log(f"[motif] {key}: {len(g['y']):,} flows, class counts {counts.tolist()}")
    return out


def survival(data, graph, fanout, n_inst=300, seed=0):
    """E3a / Proposition 1: probability that every edge of a planted cycle that contains the
    target ends up in the computation graph, empirically and as predicted.

    Prediction (endpoints of each motif edge in the frontier; uniform sampling of B incident
    edges per node with replacement; all edges kept when the degree is <= B):
                                       p_in(a->b)  = 1 - (1 - m_ab/indeg(b))^B,
                                       p_out(a->b) = 1 - (1 - m_ab/outdeg(a))^B.
      asymmetric, strict (edge in both relations): prod p_in * p_out
      asymmetric, forward relation only:           prod p_in
      coupled (shared edge set):                   prod [1 - (1 - p_in)(1 - p_out)]
    """
    from .sampler import build_batch
    rng = np.random.default_rng(seed)
    cycles = json.load(open(os.path.join(data.dir, "cycles.json")))
    src, dst = data.src, data.dst
    eids = np.asarray(graph.eids)
    key_all = src[eids] * data.num_nodes + dst[eids]
    order = np.argsort(key_all)
    key_sorted = key_all[order]
    B = fanout[0]
    indeg, outdeg = np.asarray(graph.in_deg), np.asarray(graph.out_deg)
    mult = np.asarray(graph.mult)
    res = {k: [] for k in ("asym_strict", "asym_fwd", "coupled", "pred_asym_strict", "pred_asym_fwd", "pred_coupled")}
    cfg_s = {"fanout": fanout, "tau_s": 0.0}
    for ci in rng.permutation(len(cycles))[:n_inst]:
        c = cycles[ci]
        pairs = list(zip(c, c[1:] + c[:1]))
        copies = []
        for a, b in pairs:
            k = a * data.num_nodes + b
            lo, hi = np.searchsorted(key_sorted, k), np.searchsorted(key_sorted, k, side="right")
            copies.append(eids[order[lo:hi]])
        if any(len(x) == 0 for x in copies):
            continue
        target = np.array([rng.choice(copies[0])])
        ba = build_batch(graph, target, {"coupled": False, "force": True}, cfg_s, rng)
        bc = build_batch(graph, target, {"coupled": True, "force": True}, cfg_s, rng)
        inf = lambda ids, cp: bool(np.isin(cp, ids).any())
        res["asym_strict"].append(all(inf(ba["ef_ids"], cp) and inf(ba["er_ids"], cp) for cp in copies))
        res["asym_fwd"].append(all(inf(ba["ef_ids"], cp) for cp in copies))
        res["coupled"].append(all(inf(bc["ef_ids"], cp) for cp in copies))
        keep = lambda m_, deg: 1.0 if deg <= B else 1 - (1 - m_ / deg) ** B   # deg <= B: all edges kept
        p_in = np.array([keep(mult[cp[0]], indeg[b]) for (a, b), cp in zip(pairs, copies)])
        p_out = np.array([keep(mult[cp[0]], outdeg[a]) for (a, b), cp in zip(pairs, copies)])
        res["pred_asym_strict"].append(float(np.prod(p_in * p_out)))
        res["pred_asym_fwd"].append(float(np.prod(p_in)))
        res["pred_coupled"].append(float(np.prod(1 - (1 - p_in) * (1 - p_out))))
    return {k: float(np.mean(v)) if v else None for k, v in res.items()} | {"n": len(res["coupled"])}
