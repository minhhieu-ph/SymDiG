"""Subgraph samplers for the ablation ladder (pure numpy; runs in DataLoader workers).

Two independent switches reproduce every rung of the ladder:

  coupled=False  Multi-GNN / LinkNeighborLoader behaviour on a HeteroData with relations
                 ``to`` and ``rev_to``: for every frontier node, in-edges (relation ``to``) and
                 out-edges (relation ``rev_to``) are sampled independently and kept as two
                 separate edge sets, so an edge and its reverse twin are jointly present only
                 by chance.
  coupled=True   one edge set is shared by both directions; the reverse relation is its
                 exact flip (every sampled edge has its twin).

  force=False    target (seed) edges enter the computation graph only if the neighbour
                 sampler happens to draw them (LinkNeighborLoader does not inject the
                 supervision edges into ``edge_index``); absent targets cannot be scored.
  force=True     target edges are always part of the computation graph.

Sampling of the incident edges of a node is uniform with replacement (duplicates removed),
which approximates PyG's without-replacement sampling when degree >> fan-out.
"""
import numpy as np

EMPTY = np.zeros(0, np.int64)


def _gather_all(ptr, eid, nodes):
    starts = np.asarray(ptr[nodes])
    lens = np.asarray(ptr[nodes + 1]) - starts
    tot = int(lens.sum())
    if tot == 0:
        return EMPTY
    offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return np.asarray(eid[offs + np.arange(tot)])


def sample_incident(ptr, eid, mult, nodes, B, tau, oversample, rng):
    """Up to B incident edges per node. tau > 0: P(e) ∝ M_uv^-tau (Gumbel top-B)."""
    if len(nodes) == 0 or B <= 0:
        return EMPTY
    deg = np.asarray(ptr[nodes + 1]) - np.asarray(ptr[nodes])
    small, large = nodes[(deg > 0) & (deg <= B)], nodes[deg > B]
    out = [_gather_all(ptr, eid, small)]
    if len(large):
        dl = deg[deg > B]
        c = max(B, int(oversample * B)) if tau > 0 else B
        pos = np.asarray(ptr[large])[:, None] + (rng.random((len(large), c)) * dl[:, None]).astype(np.int64)
        cand = np.asarray(eid[pos.ravel()]).reshape(pos.shape)
        if tau > 0 and c > B:
            key = -tau * np.log(np.asarray(mult[cand.ravel()], np.float64).reshape(cand.shape)) + rng.gumbel(size=cand.shape)
            top = np.argpartition(-key, B - 1, axis=1)[:, :B]
            cand = np.take_along_axis(cand, top, axis=1)
        out.append(cand.ravel())
    return np.unique(np.concatenate(out))


def expand(graph, targets, fanout, tau, oversample, max_hop_edges, rng):
    """Bidirectional neighbour expansion from the endpoints of the targets.
    Returns (sampled in-edges, sampled out-edges) over all hops."""
    src, dst = graph.src, graph.dst
    seeds = np.unique(np.concatenate([src[targets], dst[targets]]))
    frontier, visited = seeds, seeds
    e_in, e_out = [], []
    for B in fanout:
        a = sample_incident(graph.in_ptr, graph.in_eid, graph.mult, frontier, B, tau, oversample, rng)
        b = sample_incident(graph.out_ptr, graph.out_eid, graph.mult, frontier, B, tau, oversample, rng)
        if max_hop_edges and len(a) + len(b) > max_hop_edges:
            ra = max_hop_edges * len(a) // (len(a) + len(b))
            a = np.sort(rng.choice(a, ra, replace=False))
            b = np.sort(rng.choice(b, max_hop_edges - ra, replace=False))
        e_in.append(a)
        e_out.append(b)
        if len(a) + len(b) == 0:
            break
        nb = np.unique(np.concatenate([src[a], dst[a], src[b], dst[b]]))
        frontier = np.setdiff1d(nb, visited, assume_unique=True)
        visited = np.union1d(visited, nb)
        if len(frontier) == 0:
            break
    cat = lambda l: np.unique(np.concatenate(l)) if l else EMPTY
    return cat(e_in), cat(e_out)


def build_batch(graph, targets, mode, cfg_s, rng):
    """Computation subgraph for one mini-batch.

    mode: {"coupled": bool, "force": bool}
    Returns local-index arrays:
      nodes            global node ids of V_S (sorted)
      ei_f, ef_ids     forward relation edges (u -> v) and their global edge ids
      ei_r, er_ids     reverse relation: message v -> u for each edge in er_ids
      shared           True when er_ids == ef_ids (coupled)
      targets          the requested targets (sorted, unique)
      present          bool mask over targets: target is in ef_ids (scorable)
      tpos             positions in ef_ids of the present targets
      ego              local node indices of the endpoints of present targets
    """
    targets = np.unique(np.asarray(targets, np.int64))
    src, dst = graph.src, graph.dst
    e_in, e_out = expand(graph, targets, cfg_s["fanout"], cfg_s.get("tau_s", 0.0),
                         cfg_s.get("oversample", 4), cfg_s.get("max_hop_edges"), rng)
    if mode["coupled"]:
        ef = np.union1d(e_in, e_out)
        if mode["force"]:
            ef = np.union1d(ef, targets)
        er = ef
    else:
        ef = np.union1d(e_in, targets) if mode["force"] else e_in
        er = e_out
    present = np.isin(targets, ef, assume_unique=True)
    nodes = np.unique(np.concatenate([src[ef], dst[ef], src[er], dst[er], src[targets], dst[targets]]))
    loc = lambda g: np.searchsorted(nodes, g)
    ei_f = np.stack([loc(src[ef]), loc(dst[ef])])
    if mode["coupled"]:
        ei_r = ei_f[::-1].copy()
    else:
        ei_r = np.stack([loc(dst[er]), loc(src[er])])
    pt = targets[present]
    tpos = np.searchsorted(ef, pt)
    ego = np.unique(np.concatenate([loc(src[pt]), loc(dst[pt])]))
    return {"nodes": nodes, "ei_f": ei_f, "ef_ids": ef, "ei_r": ei_r, "er_ids": er,
            "shared": bool(mode["coupled"]), "targets": targets, "present": present, "tpos": tpos,
            "ego": ego}


def twin_coverage(b):
    """Fraction of forward edges whose reverse twin is also in the computation graph."""
    if b["shared"]:
        return 1.0
    return float(np.isin(b["ef_ids"], b["er_ids"]).mean()) if len(b["ef_ids"]) else 0.0


def coverage_stats(graph, targets, mode, cfg_s, batch_size, n_batches, seed):
    """E1: seed coverage and twin coverage of a sampler on real data (no training).
    Also bins coverage by the multigraph degree of the target's destination."""
    rng = np.random.default_rng(seed)
    t = rng.permutation(np.asarray(targets, np.int64))
    pres, twins, nedges = [], [], []
    deg_t, pres_all = [], []
    for i in range(n_batches):
        chunk = t[i * batch_size:(i + 1) * batch_size]
        if len(chunk) == 0:
            break
        b = build_batch(graph, chunk, mode, cfg_s, rng)
        pres.append(b["present"].mean())
        twins.append(twin_coverage(b))
        nedges.append(len(b["ef_ids"]) + (0 if b["shared"] else len(b["er_ids"])))
        deg_t.append(np.asarray(graph.in_deg)[graph.dst[b["targets"]]] + np.asarray(graph.out_deg)[graph.src[b["targets"]]])
        pres_all.append(b["present"])
    deg_t, pres_all = np.concatenate(deg_t), np.concatenate(pres_all)
    bins = [0, 10, 100, 1_000, 10_000, 100_000, np.inf]
    by_deg = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (deg_t >= lo) & (deg_t < hi)
        if m.any():
            by_deg.append({"lo": lo, "hi": None if hi == np.inf else hi, "n": int(m.sum()),
                           "coverage": float(pres_all[m].mean())})
    return {"seed_coverage": float(np.mean(pres)), "twin_coverage": float(np.mean(twins)),
            "edges_per_batch": float(np.mean(nedges)), "by_degree": by_deg, "n_targets": int(len(pres_all))}
