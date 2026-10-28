"""E7: traffic-injection evasion against graph-based detectors.

For every test flow of the attacked classes (attacker u -> victim v), the adversary injects
K benign-looking chaff flows. Chaff features are copies of randomly drawn benign test flows
(including their ports), so each chaff flow is indistinguishable from real benign traffic on
a per-flow basis. Chaff flows enter the message-passing graph but are never scored.

  pair     K chaff flows on the same pair u -> v (raises multiplicity, dilutes the pair)
  reverse  K chaff flows v -> u (targets the reverse stream / reverse relation)
  flood    K chaff flows u -> random benign hosts and K flows random hosts -> v
           (inflates both endpoints' degree so the neighbour sampler sees mostly chaff)
"""
import numpy as np

from .data import FeatureStore, Graph


def inject(data, store, base_eids, attack_targets, strategy, k, seed=0):
    """Returns (graph over base + chaff edges, feature store with the chaff rows appended)."""
    if k <= 0 or len(attack_targets) == 0:
        return Graph.build(data.src, data.dst, base_eids, data.num_nodes), store
    rng = np.random.default_rng(seed)
    u, v = data.src[attack_targets], data.dst[attack_targets]
    if strategy == "pair":
        cs, cd = np.repeat(u, k), np.repeat(v, k)
    elif strategy == "reverse":
        cs, cd = np.repeat(v, k), np.repeat(u, k)
    elif strategy == "flood":
        ben_hosts = np.unique(np.concatenate([data.src[base_eids][data.y_multi[base_eids] == 0],
                                              data.dst[base_eids][data.y_multi[base_eids] == 0]]))
        r1 = rng.choice(ben_hosts, len(u) * k)
        r2 = rng.choice(ben_hosts, len(u) * k)
        cs = np.concatenate([np.repeat(u, k), r2])
        cd = np.concatenate([r1, np.repeat(v, k)])
    else:
        raise ValueError(strategy)
    benign = base_eids[data.y_multi[base_eids] == 0]
    donors = rng.choice(benign, len(cs), replace=True)
    extra = store.raw(donors)
    E = len(data.src)
    src = np.concatenate([data.src, cs.astype(np.int64)])
    dst = np.concatenate([data.dst, cd.astype(np.int64)])
    eids = np.concatenate([base_eids, E + np.arange(len(cs))])
    g = Graph.build(src, dst, eids, data.num_nodes)
    return g, store.with_extra(extra)
