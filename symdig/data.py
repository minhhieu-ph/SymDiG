"""Data layer: CSV -> cache, features, splits, graph structure.

Cache layout (cache/<dataset>/):
  X.npy           float32 [E, d]  log1p(numeric NetFlow columns) + TCP-flag bits (no IPs, no ports)
  sport.npy dport.npy  int32 [E]  L4 ports (kept apart: "ports" is an experimental variable)
  y.npy  ybin.npy                  multiclass (Benign = 0) / binary labels
  src_ip.npy dst_ip.npy            IP-centric node ids
  src_ipport.npy dst_ipport.npy    IP:PORT node ids (E-GraphSAGE / deployability)
  meta.json
  splits/<name>.npz(+.json)        cached splits (fixed split seed)
  graphs/<...>.npz                 cached CSR structures (memory-mapped, shared by processes)

Splits (all return a Split):
  p1A  random stratified 75/25; val = val_frac of train; inductive graphs
  p1B  host-disjoint: attacker source IPs in test never initiate a training flow
  p2   FeCoGraph protocol: 2% stratified subsample, 30% labeled, 10% val, rest test; transductive
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

CACHE_VERSION = 2

DATASETS = {
    "bot": {"patterns": ["bot-iot"], "name": "NF-BoT-IoT-v2"},
    "ton": {"patterns": ["ton-iot"], "name": "NF-ToN-IoT-v2"},
    "ids2018": {"patterns": ["cse-cic-ids2018", "ids2018"], "name": "NF-CSE-CIC-IDS2018-v2"},
    "unsw": {"patterns": ["unsw-nb15", "unsw"], "name": "NF-UNSW-NB15-v2"},
}

IP_SRC, IP_DST = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
P_SRC, P_DST = "L4_SRC_PORT", "L4_DST_PORT"
FLAG_COLS = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
LABEL_COLS = ["Label", "Attack", "Dataset"]
CANONICAL = {
    "benign": "Benign", "scanning": "Scanning", "xss": "XSS", "ddos": "DDoS", "dos": "DoS",
    "password": "Password", "injection": "Injection", "backdoor": "Backdoor", "mitm": "MITM",
    "ransomware": "Ransomware", "theft": "Theft", "reconnaissance": "Reconnaissance",
    "exploits": "Exploits", "fuzzers": "Fuzzers", "generic": "Generic", "analysis": "Analysis",
    "shellcode": "Shellcode", "worms": "Worms",
}


def dataset_name(key):
    if key in DATASETS:
        return DATASETS[key]["name"]
    return key


def group_ids2018(name):
    """NF-CSE-CIC-IDS2018-v2 sub-attacks -> 6 families (+ Benign)."""
    n = name.lower()
    if n == "benign":
        return "Benign"
    if "ddos" in n:
        return "DDoS"
    if "dos" in n:
        return "DoS"
    if "web" in n or "xss" in n or "sql" in n:
        return "Web"
    if "brute" in n or "ftp" in n or "ssh" in n:
        return "BruteForce"
    if "bot" in n:
        return "Bot"
    if "infil" in n:
        return "Infiltration"
    return name


def canonical_attack(name, dataset, group=True):
    name = str(name).strip()
    if dataset == "ids2018" and group:
        return group_ids2018(name)
    return CANONICAL.get(name.lower(), name)


def find_csv(root, key):
    pats = DATASETS[key]["patterns"]
    files = sorted(set(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)))
    hits = [f for f in files if any(p in os.path.basename(f).lower() for p in pats)]
    if not hits:
        raise FileNotFoundError(f"No CSV for '{key}' under {root} (looked for {pats}). Found: {files}")
    hits.sort(key=lambda f: ("v2" not in f.lower(), "test" in f.lower() or "sample" in f.lower()))
    return hits[0]


def _atomic_save(path, arr):
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


class _KeyEncoder:
    def __init__(self):
        self.table = {}

    def __call__(self, keys: pd.Series) -> np.ndarray:
        for k in pd.unique(keys):
            if k not in self.table:
                self.table[k] = len(self.table)
        return keys.map(self.table).to_numpy(np.int64).astype(np.int32)


def _flag_bits(v):
    v = v.astype(np.int64) & 0xFF
    return ((v[:, None] >> np.arange(8)) & 1).astype(np.float32)


def cache_dir(cfg, key):
    root = cfg["paths"]["cache"]
    n = cfg["data"].get("max_rows")
    return os.path.join(root, key if not n else f"{key}_n{n}")


def is_ready(cfg, key):
    return os.path.exists(os.path.join(cache_dir(cfg, key), "meta.json"))


def prepare(cfg, key, chunksize=2_000_000, log=print):
    """Read the CSV once (chunked) and write the cache. Idempotent and atomic."""
    if key.startswith("motif"):
        from .motifs import prepare_motif
        return prepare_motif(cfg, key, log=log)
    out = cache_dir(cfg, key)
    if is_ready(cfg, key):
        meta = json.load(open(os.path.join(out, "meta.json")))
        if meta.get("version") != CACHE_VERSION:
            raise RuntimeError(f"{out} was written by an older version; delete it and re-run prepare")
        return out
    os.makedirs(out, exist_ok=True)
    path = find_csv(cfg["paths"]["data_root"], key)
    log(f"[prepare] {key}: reading {path}")
    header = pd.read_csv(path, nrows=1000)
    missing = [c for c in [IP_SRC, IP_DST, P_SRC, P_DST, "Label", "Attack"] if c not in header.columns]
    if missing:
        raise ValueError(f"{path} lacks NetFlow v2 columns {missing}")
    skip = set([IP_SRC, IP_DST, P_SRC, P_DST] + FLAG_COLS + LABEL_COLS)
    num_cols = [c for c in header.columns if c not in skip and pd.api.types.is_numeric_dtype(header[c])]
    flag_cols = [c for c in FLAG_COLS if c in header.columns]
    feat_names = num_cols + [f"{c}_b{i}" for c in flag_cols for i in range(8)]

    enc_ip, enc_ipport, codes = _KeyEncoder(), _KeyEncoder(), {}
    tmp, n_rows, t0 = [], 0, time.time()
    max_rows = cfg["data"].get("max_rows")
    group = cfg["data"].get("group_classes", True)
    reader = pd.read_csv(path, chunksize=chunksize, dtype={IP_SRC: str, IP_DST: str}, low_memory=False)
    for ci, df in enumerate(reader):
        if max_rows and n_rows >= max_rows:
            break
        if max_rows:
            df = df.iloc[: max_rows - n_rows]
        sp = df[P_SRC].fillna(0).astype(np.int64)
        dp = df[P_DST].fillna(0).astype(np.int64)
        s_ip, d_ip = df[IP_SRC].astype(str), df[IP_DST].astype(str)
        num = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        num = np.log1p(np.clip(num, 0, None)).astype(np.float32)
        X = np.concatenate([num] + [_flag_bits(df[c].fillna(0).to_numpy()) for c in flag_cols], 1)
        att = df["Attack"].map(lambda a: canonical_attack(a, key, group))
        for a in pd.unique(att):
            codes.setdefault(a, len(codes))
        arrays = {
            "X": X, "y_raw": att.map(codes).to_numpy(np.int16), "ybin": df["Label"].to_numpy().astype(np.int8),
            "sport": sp.to_numpy(np.int32), "dport": dp.to_numpy(np.int32),
            "src_ip": enc_ip(s_ip), "dst_ip": enc_ip(d_ip),
            "src_ipport": enc_ipport(s_ip + ":" + sp.astype(str)),
            "dst_ipport": enc_ipport(d_ip + ":" + dp.astype(str)),
        }
        f = os.path.join(out, f"_chunk{ci:04d}.npz")
        np.savez(f, **arrays)
        tmp.append((f, len(df)))
        n_rows += len(df)
        log(f"  chunk {ci}: {n_rows:,} rows, {len(enc_ip.table):,} IPs, {time.time() - t0:.0f}s")

    counts = np.zeros(len(codes), np.int64)
    for f, _ in tmp:
        counts += np.bincount(np.load(f)["y_raw"], minlength=len(codes))
    names = list(codes.keys())
    order = sorted(range(len(names)), key=lambda i: (names[i] != "Benign", -counts[i]))
    remap = np.empty(len(names), np.int16)
    remap[order] = np.arange(len(names))
    classes = [names[i] for i in order]

    spec = {"X": (np.float32, (n_rows, len(feat_names))), "y": (np.int16, (n_rows,)), "ybin": (np.int8, (n_rows,))}
    for k in ("sport", "dport", "src_ip", "dst_ip", "src_ipport", "dst_ipport"):
        spec[k] = (np.int32, (n_rows,))
    mm = {k: np.lib.format.open_memmap(os.path.join(out, f"{k}.npy"), mode="w+", dtype=dt, shape=sh)
          for k, (dt, sh) in spec.items()}
    pos = 0
    for f, n in tmp:
        z = np.load(f)
        mm["X"][pos:pos + n] = z["X"]
        mm["y"][pos:pos + n] = remap[z["y_raw"]]
        for k in ("ybin", "sport", "dport", "src_ip", "dst_ip", "src_ipport", "dst_ipport"):
            mm[k][pos:pos + n] = z[k]
        pos += n
        os.remove(f)
    for a in mm.values():
        a.flush()
    meta = {"version": CACHE_VERSION, "dataset": key, "name": dataset_name(key), "csv": path,
            "n_edges": int(n_rows), "classes": classes, "class_counts": [int(counts[i]) for i in order],
            "num_nodes": {"ip": len(enc_ip.table), "ipport": len(enc_ipport.table)},
            "feature_names": feat_names, "has_ports": True}
    _atomic_json(os.path.join(out, "meta.json"), meta)
    log(f"[prepare] {key}: {n_rows:,} flows, {len(classes)} classes, {meta['num_nodes']['ip']:,} IPs, "
        f"{meta['num_nodes']['ipport']:,} IP:PORT")
    return out


# --------------------------------------------------------------------------- features
PORT_FEATURES = ["sport", "dport", "sport_wellknown", "sport_registered", "sport_dynamic",
                 "dport_wellknown", "dport_registered", "dport_dynamic"]


def _port_block(sp, dp):
    sp = np.asarray(sp, np.int64)
    dp = np.asarray(dp, np.int64)
    oh = lambda p: np.stack([p < 1024, (p >= 1024) & (p < 49152), p >= 49152], 1)
    return np.concatenate([np.stack([sp, dp], 1), oh(sp), oh(dp)], 1).astype(np.float32)


class FeatureStore:
    """Edge features gathered on demand, standardised with train statistics.
    Optional port features; optional appended rows (evasion chaff)."""

    def __init__(self, X, sport, dport, ports, mean, std, extra=None):
        self.X, self.sport, self.dport, self.ports = X, sport, dport, ports
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.extra = extra
        self.n_base = X.shape[0]
        self.dim = X.shape[1] + (len(PORT_FEATURES) if ports else 0)

    def raw(self, ids):
        ids = np.asarray(ids, np.int64)
        order = np.argsort(ids, kind="stable")  # sorted reads are much faster on memmaps
        sids = ids[order]
        base = sids < self.n_base
        out = np.empty((len(ids), self.dim), np.float32)
        d0 = self.X.shape[1]
        b = sids[base]
        tmp = np.empty((len(sids), self.dim), np.float32)
        tmp[base, :d0] = self.X[b]
        if self.ports:
            tmp[base, d0:] = _port_block(self.sport[b], self.dport[b])
        if (~base).any():
            tmp[~base] = self.extra[sids[~base] - self.n_base]
        out[order] = tmp
        return out

    def gather(self, ids):
        out = self.raw(ids)
        out -= self.mean
        out /= self.std
        return out

    def with_extra(self, extra_raw):
        return FeatureStore(self.X, self.sport, self.dport, self.ports, self.mean, self.std,
                            np.asarray(extra_raw, np.float32))

    def state(self):
        return {"mean": self.mean, "std": self.std, "ports": self.ports}


class NetFlowData:
    """Arrays of one cached dataset (memory-mapped by default: shared page cache
    across the worker processes of one machine)."""

    def __init__(self, cfg, key, task="multi", node_mode="ip", log=print):
        if not is_ready(cfg, key):
            prepare(cfg, key, log=log)
        d = cache_dir(cfg, key)
        self.cfg, self.key, self.task, self.node_mode, self.dir = cfg, key, task, node_mode, d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        mm = "r" if not cfg["data"].get("in_memory", False) else None
        ld = lambda n: np.load(os.path.join(d, f"{n}.npy"), mmap_mode=mm)
        self.X = ld("X")
        self.y_multi = np.asarray(ld("y")).astype(np.int64)
        self.y_bin = np.asarray(ld("ybin")).astype(np.int64)
        self.has_ports = bool(self.meta.get("has_ports", False))
        self.sport = ld("sport") if self.has_ports else None
        self.dport = ld("dport") if self.has_ports else None
        self.src = np.asarray(ld(f"src_{node_mode}")).astype(np.int64)
        self.dst = np.asarray(ld(f"dst_{node_mode}")).astype(np.int64)
        self.num_nodes = int(self.meta["num_nodes"][node_mode])
        if task == "binary":
            self.y, self.classes = self.y_bin, ["Benign", "Attack"]
        else:
            self.y, self.classes = self.y_multi, list(self.meta["classes"])
        self.n_classes = len(self.classes)
        self.n_edges = len(self.y)

    def ip_arrays(self):
        ld = lambda n: np.asarray(np.load(os.path.join(self.dir, f"{n}.npy"), mmap_mode="r")).astype(np.int64)
        return ld("src_ip"), ld("dst_ip")

    def make_store(self, fit_idx, ports, seed=0, max_fit=2_000_000):
        ports = bool(ports) and self.has_ports
        rng = np.random.default_rng(seed)
        fit = np.sort(fit_idx if len(fit_idx) <= max_fit else rng.choice(fit_idx, max_fit, replace=False))
        tmp = FeatureStore(self.X, self.sport, self.dport, ports, np.zeros(1), np.ones(1))
        R = tmp.raw(fit).astype(np.float64)
        mean, std = R.mean(0), R.std(0)
        std[std < 1e-6] = 1.0
        return FeatureStore(self.X, self.sport, self.dport, ports, mean, std)

    def feature_names(self, ports):
        return list(self.meta["feature_names"]) + (PORT_FEATURES if ports and self.has_ports else [])


# --------------------------------------------------------------------------- splits
@dataclass
class Split:
    name: str
    train: np.ndarray        # labeled training targets
    val: np.ndarray          # validation targets (model selection)
    test: np.ndarray         # test targets
    graph_train: np.ndarray  # edges available for message passing during training / validation
    graph_test: np.ndarray   # edges available for message passing at test time
    meta: dict


def _strat(y, idx, frac, rng, min_keep=1):
    """Per-class random subset of `idx` of size round(frac * n_c) (>= 1 when possible)."""
    out = []
    yy = y[idx]
    for c in np.unique(yy):
        ci = idx[yy == c]
        k = int(round(len(ci) * frac))
        if len(ci) > min_keep:
            k = min(max(k, 1), len(ci) - min_keep) if frac > 0 else 0
        else:
            k = 0
        out.append(rng.choice(ci, k, replace=False) if k else ci[:0])
    return np.sort(np.concatenate(out)) if out else idx[:0]


def _val_from(y, train_all, cfg, rng):
    dcfg = cfg["data"]
    n_val = min(int(dcfg.get("val_max", 200_000)), int(len(train_all) * dcfg.get("val_frac", 0.1)))
    val = _strat(y, train_all, n_val / max(len(train_all), 1), rng) if n_val > 0 else train_all[:0]
    return val, np.setdiff1d(train_all, val, assume_unique=True)


def host_split(y, src_ip, test_frac, rng, min_ips=2):
    """Attacker-source-disjoint split.

    Test IP set T is grown class by class (rarest attack class first): source IPs of that
    class are added in random order until ~test_frac of the class's flows originate in T.
    A flow is in test iff its source IP is in T. Benign flows from IPs outside T are split
    at random so that benign also has ~test_frac in test. Classes with fewer than
    `min_ips` distinct source IPs cannot be split; they are kept in train only and listed
    as excluded."""
    n = len(y)
    classes = [c for c in np.unique(y) if c != 0]
    counts = {c: int((y == c).sum()) for c in classes}
    T = set()
    stats, excluded = {}, []
    for c in sorted(classes, key=lambda c: counts[c]):
        m = y == c
        ips, cnt = np.unique(src_ip[m], return_counts=True)
        if len(ips) < min_ips:
            excluded.append(int(c))
            stats[int(c)] = {"src_ips": int(len(ips)), "flows": counts[c], "excluded": True}
            continue
        in_t = np.array([ip in T for ip in ips])
        share = cnt[in_t].sum() / cnt.sum()
        for j in rng.permutation(np.flatnonzero(~in_t)):
            if share >= test_frac:
                break
            if cnt[j] / cnt.sum() > 0.9 and share > 0:  # never let a single giant IP flip the class
                continue
            T.add(int(ips[j]))
            share += cnt[j] / cnt.sum()
        stats[int(c)] = {"src_ips": int(len(ips)), "flows": counts[c], "excluded": False}
    T_arr = np.array(sorted(T), np.int64)
    in_test = np.isin(src_ip, T_arr)
    dropped = np.zeros(n, bool)
    for c in excluded:
        # unsplittable class: kept in train, except flows from test IPs, which are dropped from
        # both parts (they would otherwise leak a test attacker into the training targets)
        m = y == c
        dropped |= m & in_test
        in_test[m] = False
    ben = np.flatnonzero((y == 0) & ~in_test)
    need = max(0, int(test_frac * (y == 0).sum()) - int(((y == 0) & in_test).sum()))
    if need:
        in_test[rng.choice(ben, min(need, len(ben)), replace=False)] = True
    test = np.flatnonzero(in_test)
    train = np.flatnonzero(~in_test & ~dropped)
    for c in list(stats):
        m = y == c
        stats[c]["test_frac"] = float(in_test[m].mean())
        stats[c]["test_src_ips"] = int(len(np.intersect1d(np.unique(src_ip[m & in_test]), T_arr)))
    stats[0] = {"test_frac": float(in_test[y == 0].mean())}
    # leakage check: no test attacker IP initiates an attack flow in train
    att_train_src = np.unique(src_ip[(~in_test) & (~dropped) & (y != 0)])
    leak = len(np.intersect1d(att_train_src, T_arr))
    return train, test, {"per_class": {str(k): v for k, v in stats.items()}, "excluded_classes": excluded,
                         "n_test_ips": int(len(T_arr)), "attack_ip_overlap": int(leak),
                         "dropped_flows": int(dropped.sum())}


def get_split(cfg, data, name):
    """Cached split. `name` in {p1A, p1B, p2}. The split seed is fixed in the config so that
    every model and every model seed sees the same split."""
    d = os.path.join(data.dir, "splits")
    os.makedirs(d, exist_ok=True)
    seed = int(cfg["data"].get("split_seed", 42))
    tag = f"{name}_s{seed}"
    if name == "p2":
        p2 = cfg["protocols"]["p2"]
        tag += f"_f{p2['sample_frac']}_l{p2['label_frac']}"
    f = os.path.join(d, f"{tag}.npz")
    if os.path.exists(f):
        z = np.load(f)
        meta = json.load(open(f.replace(".npz", ".json")))
        return Split(name, z["train"], z["val"], z["test"], z["graph_train"], z["graph_test"], meta)

    y = data.y_multi
    rng = np.random.default_rng(seed)
    allidx = np.arange(len(y))
    ts = float(cfg["data"].get("test_size", 0.25))
    if name == "p1A":
        test = _strat(y, allidx, ts, rng)
        train_all = np.setdiff1d(allidx, test, assume_unique=True)
        val, train = _val_from(y, train_all, cfg, rng)
        meta = {}
        gtr, gte = train_all, allidx
    elif name == "p1B":
        src_ip, _ = data.ip_arrays()
        train_all, test, meta = host_split(y, src_ip, ts, rng)
        val, train = _val_from(y, train_all, cfg, rng)
        gtr, gte = train_all, allidx
    elif name == "p2":
        p2 = cfg["protocols"]["p2"]
        sub = _strat(y, allidx, float(p2["sample_frac"]), rng, min_keep=0)
        train = _strat(y, sub, float(p2["label_frac"]), rng, min_keep=2)
        rest = np.setdiff1d(sub, train, assume_unique=True)
        vf = min(float(p2["val_frac"]) * len(sub) / max(len(rest), 1), 0.9)
        val = _strat(y, rest, vf, rng, min_keep=1)
        test = np.setdiff1d(rest, val, assume_unique=True)
        meta = {"subset_size": int(len(sub))}
        gtr = gte = sub
    else:
        raise ValueError(name)
    meta.update({"n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
                 "test_class_counts": np.bincount(y[test], minlength=len(data.meta["classes"])).tolist(),
                 "train_class_counts": np.bincount(y[train], minlength=len(data.meta["classes"])).tolist()})
    tmp = f"{f}.tmp{os.getpid()}.npz"
    np.savez(tmp, train=train, val=val, test=test, graph_train=gtr, graph_test=gte)
    os.replace(tmp, f)
    _atomic_json(f.replace(".npz", ".json"), meta)
    return Split(name, train, val, test, gtr, gte, meta)


def subset_split(split, frac, y, seed):
    """Stratified fraction of every part of a split (data-scale experiments)."""
    rng = np.random.default_rng(seed)
    keep = _strat(y, np.arange(len(y)), frac, rng, min_keep=0)
    f = lambda a: np.intersect1d(a, keep, assume_unique=True)
    return Split(f"{split.name}_frac{frac}", f(split.train), f(split.val), f(split.test),
                 f(split.graph_train), f(split.graph_test), dict(split.meta, frac=frac))


# --------------------------------------------------------------------------- graph structure
class Graph:
    """CSR view of the directed multigraph induced by a subset of edges (global edge ids)."""

    ARRAYS = ("eids", "out_eid", "out_ptr", "in_eid", "in_ptr", "mult")

    def __init__(self, src, dst, num_nodes, arrays):
        self.src, self.dst, self.num_nodes = src, dst, num_nodes
        for k in self.ARRAYS:
            setattr(self, k, arrays[k])
        self.out_deg = np.diff(self.out_ptr)
        self.in_deg = np.diff(self.in_ptr)

    @classmethod
    def build(cls, src, dst, eids, num_nodes):
        eids = np.sort(np.asarray(eids, np.int64))
        s, t = src[eids], dst[eids]
        o = np.argsort(s, kind="stable")
        out_ptr = np.zeros(num_nodes + 1, np.int64)
        np.cumsum(np.bincount(s, minlength=num_nodes), out=out_ptr[1:])
        i = np.argsort(t, kind="stable")
        in_ptr = np.zeros(num_nodes + 1, np.int64)
        np.cumsum(np.bincount(t, minlength=num_nodes), out=in_ptr[1:])
        key = s * np.int64(num_nodes) + t
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        mult = np.zeros(len(src), np.int32)
        mult[eids] = cnt[inv.ravel()]
        return cls(src, dst, num_nodes, {"eids": eids, "out_eid": eids[o], "out_ptr": out_ptr,
                                         "in_eid": eids[i], "in_ptr": in_ptr, "mult": mult})

    @classmethod
    def cached(cls, data, eids, tag):
        """Build once per (dataset, split, node mode); later loads are memory-mapped."""
        d = os.path.join(data.dir, "graphs")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, f"{tag}_{data.node_mode}")
        if all(os.path.exists(f"{base}.{k}.npy") for k in cls.ARRAYS):
            arrs = {k: np.load(f"{base}.{k}.npy", mmap_mode="r") for k in cls.ARRAYS}
            return cls(data.src, data.dst, data.num_nodes, arrs)
        g = cls.build(data.src, data.dst, eids, data.num_nodes)
        for k in cls.ARRAYS:
            _atomic_save(f"{base}.{k}.npy", getattr(g, k))
        return g

    @property
    def n_edges(self):
        return len(self.eids)

    def stats(self):
        deg = (self.out_deg + self.in_deg).astype(np.float64)
        e = np.asarray(self.eids)
        return {"nodes": int((deg > 0).sum()), "edges": int(len(e)),
                "line_graph_edges": float((deg * (deg - 1) / 2).sum()),
                "max_degree": int(deg.max()) if len(deg) else 0,
                "mean_multiplicity": float(np.asarray(self.mult)[e].mean()) if len(e) else 0.0}


def graph_for(data, split, which):
    eids = split.graph_train if which == "train" else split.graph_test
    if split.name.startswith(("p1A", "p1B", "p2")) and "frac" not in split.name:
        return Graph.cached(data, eids, f"{split.name}_{which}")
    return Graph.build(data.src, data.dst, eids, data.num_nodes)


def line_graph_size(src, dst):
    """Edges of the line graph of a directed multigraph (two flows adjacent iff they share a host)."""
    nodes = np.concatenate([src, dst])
    flows = np.concatenate([np.arange(len(src)), np.arange(len(src))])
    key = np.unique(nodes.astype(np.int64) * len(src) + flows)  # a self-loop flow counts once per host
    _, deg = np.unique(key // len(src), return_counts=True)
    deg = deg.astype(np.float64)
    return float((deg * (deg - 1) / 2).sum())
