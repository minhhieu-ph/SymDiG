"""Expands each experiment of the proposal into job specs.

E1  diagnosis: seed/twin coverage of the samplers (no training) + Multi-GNN "as released"
E2  ablation ladder 0-4 on the four datasets
E3  synthetic motifs: survival vs multiplicity (empirical vs Proposition 1) + ladder 0-4
E4  same-frame comparison: tuned baselines on p1A (+ ports ablation) and on the FeCoGraph protocol p2
E5  robustness: host-disjoint split p1B + cross-dataset binary transfer
E6  deployability: graph sizes (IP vs IP:PORT vs line graph), VRAM / throughput / latency, 16 GB cap
E7  adaptive traffic injection (pair / reverse / flood) against trained detectors
"""
import copy

from .jobs import job_id

PRIO = {"prepare": 0, "coverage": 10, "survival": 10, "sizes": 10, "tune": 20, "gnn": 40, "tabular": 40,
        "fecograph": 45, "transfer": 60, "deploy": 70, "evasion": 80}


def spec(exp, kind, dataset, after=(), **kw):
    s = {"exp": exp, "kind": kind, "dataset": dataset, "after": [a for a in after if a],
         "priority": PRIO.get(kind, 50), "cfg_hash": Planner.CFG_HASH}
    s.update(kw)
    return s


class Planner:
    CFG_HASH = None

    def __init__(self, cfg, datasets=None, seeds=None, cfg_hash=None):
        Planner.CFG_HASH = cfg_hash
        self.cfg = cfg
        self.e = cfg["experiments"]
        self.datasets = datasets
        self.seeds_override = seeds
        self.specs = []

    # --------------------------------------------------------------- helpers
    def ds(self, default):
        return list(self.datasets) if self.datasets else list(default)

    def seeds(self, dataset):
        if self.seeds_override is not None:
            return list(self.seeds_override)
        return list(self.e["seeds_large"] if dataset == "bot" else self.e["seeds"])

    def add(self, s):
        self.specs.append(s)
        return job_id(s)

    def prep(self, dataset):
        return self.add(spec("prepare", "prepare", dataset))

    def tune(self, model, dataset):
        if self.cfg["models"][model]["kind"] == "fecograph":
            return None
        kind = self.cfg["models"][model]["kind"]
        td = self.cfg["tuning"]["gnn_datasets" if kind == "gnn" else "datasets"]
        tds = dataset if dataset in td else td[0]
        return self.add(spec("tune", "tune", tds, after=[self.prep(tds)], model=model, split="p1A",
                             task="multi", trials=int(self.cfg["tuning"]["trials"]))), f"{model}__{tds}"

    def final(self, exp, model, dataset, split, seed, task="multi", overrides=None, tuned=True):
        kind = self.cfg["models"][model]["kind"]
        after = [self.prep(dataset)]
        tkey = None
        if tuned and kind != "fecograph":
            tid, tkey = self.tune(model, dataset)
            after.append(tid)
        s = spec(exp, kind, dataset, after=after, model=model, split=split, task=task, seed=seed,
                 tuned=tkey, overrides=overrides or {})
        return self.add(s)

    # --------------------------------------------------------------- experiments
    def E1(self):
        for d in self.ds(self.e["datasets"]):
            p = self.prep(d)
            for fo in self.e["e1_fanouts"]:
                self.add(spec("E1", "coverage", d, after=[p], split="p1A", fanout=fo,
                              batches=int(self.e["e1_batches"])))
            for s in self.seeds(d):
                self.add(spec("E1", "gnn", d, after=[p], model="ladder0", split="p1A", task="multi", seed=s,
                              preset="multignn_paper", overrides={}))

    def E2(self):
        for d in self.ds(self.e["datasets"]):
            p = self.prep(d)
            for rung in range(5):
                for s in self.seeds(d):
                    self.add(spec("E2", "gnn", d, after=[p], model=f"ladder{rung}", split="p1A", task="multi",
                                  seed=s, overrides={}))

    def E3(self):
        for M in self.cfg["motif"]["multiplicities"]:
            d = f"motif_m{M}"
            p = self.prep(d)
            self.add(spec("E3", "survival", d, after=[p], split="p1A", fanout=[20, 20]))
            for rung in range(5):
                for s in self.e["e3_seeds"]:
                    self.add(spec("E3", "gnn", d, after=[p], model=f"ladder{rung}", split="p1A", task="multi",
                                  seed=s, overrides={"data.ports": False}))

    def E4(self):
        for d in self.ds(self.e["datasets"]):
            for m in self.e["e4_models"]:
                for s in self.seeds(d):
                    self.final("E4", m, d, "p1A", s)
            for m in ("xgboost", "symdig"):
                for s in self.seeds(d):
                    self.final("E4", m, d, "p1A", s, overrides={"data.ports": False})
        for d in self.ds(self.e["e4_p2_datasets"]):
            for m in self.e["e4_p2_models"]:
                for s in self.seeds(d):
                    self.final("E4", m, d, "p2", s)

    def E5(self):
        dss = self.ds(self.e["datasets"])
        for d in dss:
            for m in self.e["e5_models"]:
                for s in self.seeds(d):
                    self.final("E5", m, d, "p1B", s)
        for m in self.e["e5_transfer_models"]:
            kind = self.cfg["models"][m]["kind"]
            for src in dss:
                for s in self.seeds(src)[:3]:
                    if kind == "gnn":
                        src_id = self.final("E5", m, src, "p1A", s, task="binary")
                    for tgt in dss:
                        if tgt == src:
                            continue
                        after = [self.prep(src), self.prep(tgt)]
                        tid, tkey = self.tune(m, src)
                        after.append(tid)
                        if kind == "gnn":
                            after.append(src_id)
                        self.add(spec("E5", "transfer", tgt, after=after, model=m, source=src, split="p1A",
                                      task="binary", seed=s, tuned=tkey,
                                      source_job=src_id if kind == "gnn" else None))

    def E6(self):
        for d in self.ds(self.e["e6_datasets"]):
            p = self.prep(d)
            self.add(spec("E6", "sizes", d, after=[p], fracs=self.e["e6_fracs"]))
            for m in self.e["e6_models"]:
                tid, tkey = self.tune(m, d)
                for cap in self.e["e6_mem_caps_gb"]:
                    self.add(spec("E6", "deploy", d, after=[p, tid], model=m, split="p1A", task="multi", seed=0,
                                  tuned=tkey, frac=1.0, mem_cap_gb=cap, steps=int(self.e["e6_profile_steps"])))
            for f in self.e["e6_fracs"]:
                if f < 1.0:
                    tid, tkey = self.tune("symdig", d)
                    self.add(spec("E6", "deploy", d, after=[p, tid], model="symdig", split="p1A", task="multi",
                                  seed=0, tuned=tkey, frac=f, mem_cap_gb=None, steps=int(self.e["e6_profile_steps"])))

    def E7(self):
        d = self.e["e7_dataset"]
        if self.datasets:
            d = self.datasets[0]
        for m in self.e["e7_models"]:
            for s in self.seeds(d)[:3]:
                src = self.final("E4", m, d, "p1A", s)
                self.add(spec("E7", "evasion", d, after=[self.prep(d), src], model=m, split="p1A", task="multi",
                              seed=s, source_job=src, classes=self.e["e7_classes"],
                              strategies=self.e["e7_strategies"], k=self.e["e7_k"]))

    def tune_all(self):
        for m in self.e["e4_models"]:
            kind = self.cfg["models"][m]["kind"]
            for d in self.ds(self.cfg["tuning"]["gnn_datasets" if kind == "gnn" else "datasets"]):
                self.tune(m, d)

    def plan(self, exps):
        for x in exps:
            if x == "tune":
                self.tune_all()
            else:
                getattr(self, x)()
        return self.specs


def smoke_config(cfg):
    """Tiny version of every experiment (minutes): use with --results results_smoke."""
    c = copy.deepcopy(cfg)
    c["data"]["max_rows"] = 200_000
    c["data"]["val_max"] = 2_000
    c["train"].update(epochs=2, max_steps_per_epoch=3, batch_size=512, eval_batch_size=4096, patience=2, log_every=1)
    c["model"]["hidden"] = 32
    c["sampler"]["fanout"] = [5, 5]
    c["multignn_paper"]["sampler"] = {"fanout": [10, 10]}
    c["multignn_paper"]["model"]["hidden"] = 32
    c["tuning"].update(trials=2, gnn_epochs=1, gnn_steps=2)
    c["motif"].update(n_nodes=500, multiplicities=[1, 16])
    for k in ("xgboost", "rf"):
        c["tabular_defaults"][k]["n_estimators"] = 20
    c["tabular_defaults"]["mlp"].update(epochs=2, steps_per_epoch=5)
    c["fecograph"].update(epochs=20, eval_every=5, patience=2)
    e = c["experiments"]
    e.update(seeds=[0], seeds_large=[0], e1_fanouts=[[5, 5]], e1_batches=3, e3_seeds=[0], e6_fracs=[0.1, 1.0],
             e6_profile_steps=3, e6_mem_caps_gb=[None], e7_k=[0, 2], e4_p2_datasets=["ton"])
    c["protocols"]["p2"]["sample_frac"] = 0.2
    return c
