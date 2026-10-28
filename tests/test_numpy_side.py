"""Tests for everything that does not need PyTorch: data layer, splits, samplers, coverage,
motif generator + survival theory, graph sizes, tabular baselines (sklearn fallback when
xgboost is absent), tuning, the job queue/planner, and the table/figure generators.

    python tests/synthetic.py --out tests/tmp_dataset --rows 80000
    python tests/test_numpy_side.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from symdig.data import Graph, NetFlowData, get_split, graph_for, line_graph_size  # noqa: E402
from symdig.jobs import Queue, job_id, load_config  # noqa: E402
from symdig.planner import Planner, smoke_config  # noqa: E402
from symdig.sampler import build_batch, coverage_stats  # noqa: E402

RES = "tests/tmp_results"
CACHE = "tests/tmp_cache"


def cfg_for_tests():
    cfg = smoke_config(load_config("configs/default.yaml"))
    cfg["paths"].update(data_root="tests/tmp_dataset", cache=CACHE, results=RES)
    cfg["data"]["max_rows"] = None
    cfg["runtime"] = {"threads": 2, "num_workers": 0}
    return cfg


def test_sampler(cfg):
    d = NetFlowData(cfg, "ton")
    sp = get_split(cfg, d, "p1A")
    g = graph_for(d, sp, "train")
    t = np.random.default_rng(0).choice(sp.train, 300, replace=False)
    s_cfg = {"fanout": [10, 10], "tau_s": 0.0, "oversample": 4, "max_hop_edges": 50000}
    for coupled in (False, True):
        for force in (False, True):
            b = build_batch(g, t, {"coupled": coupled, "force": force}, s_cfg, np.random.default_rng(1))
            nodes = b["nodes"]
            assert np.array_equal(nodes[b["ei_f"][0]], d.src[b["ef_ids"]])
            assert np.array_equal(nodes[b["ei_f"][1]], d.dst[b["ef_ids"]])
            if coupled:
                assert np.array_equal(b["ei_r"], b["ei_f"][::-1]) and b["shared"]
            else:
                assert np.array_equal(nodes[b["ei_r"][0]], d.dst[b["er_ids"]])
            if force:
                assert b["present"].all()
            assert np.array_equal(b["ef_ids"][b["tpos"]], b["targets"][b["present"]])
            assert np.isin(d.src[b["targets"][b["present"]]], nodes[b["ego"]]).all()
            print(f"  sampler coupled={coupled} force={force}: present {b['present'].mean():.3f}, |E_f|={len(b['ef_ids'])}")
    st = coverage_stats(g, sp.train, {"coupled": False, "force": False}, s_cfg, 256, 5, 0)
    st2 = coverage_stats(g, sp.train, {"coupled": True, "force": True}, s_cfg, 256, 5, 0)
    assert st2["seed_coverage"] == 1.0 and st2["twin_coverage"] == 1.0 and st["seed_coverage"] < 1.0
    print(f"  coverage multignn-style {st['seed_coverage']:.3f} twin {st['twin_coverage']:.3f}; by degree {st['by_degree'][:2]}")


def test_line_graph():
    src, dst = np.array([0, 0, 1, 2]), np.array([1, 1, 2, 2])
    # hosts: 0:{f0,f1} 1:{f0,f1,f2} 2:{f2,f3(self-loop counted once)} -> 1 + 3 + 1
    assert line_graph_size(src, dst) == 5.0


def run_kinds(cfg, kinds, max_loops=500):
    sys.argv = ["run.py"]
    import run as runmod
    q = Queue(RES)
    n = 0
    for _ in range(max_loops):
        spec, _ = q.claim(lambda s: s["kind"] in kinds)
        if spec is None:
            break
        ok = runmod.run_job(cfg, q, spec, log_to_stdout=False)
        n += 1
        if not ok:
            print(open(os.path.join(RES, "logs", spec["id"] + ".log")).read()[-3000:])
            raise SystemExit(f"job {spec['id']} ({spec['kind']}) failed")
    return n


def test_pipeline(cfg):
    shutil.rmtree(RES, ignore_errors=True)
    q = Queue(RES)
    specs = Planner(cfg, ["ton", "unsw"], [0], cfg_hash="test").plan(["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    ids = set()
    for s in specs:
        jid, _ = q.add(s)
        ids.add(jid)
    assert len(ids) == len({job_id(dict(s)) for s in specs})
    print(f"  planned {len(specs)} specs -> {len(ids)} unique jobs; kinds:",
          sorted({s['kind'] for s in specs}))
    n = run_kinds(cfg, {"prepare", "coverage", "survival", "sizes"})
    print(f"  ran {n} numpy-only jobs")
    # tabular tuning + final tabular runs + tabular transfer (sklearn fallback without xgboost)
    tab = {"xgboost", "rf"}
    for jid in q.list("pending"):
        s = q.spec(jid)
        if s["kind"] == "tune" and s["model"] in tab:
            os.replace(q.p("pending", jid), q.p("running", jid))
            import run as runmod
            assert runmod.run_job(cfg, q, s, log_to_stdout=False), open(os.path.join(RES, "logs", jid + ".log")).read()[-3000:]
    for jid in q.list("pending"):
        s = q.spec(jid)
        if (s["kind"] == "tabular" and s["model"] in tab) or (s["kind"] == "transfer" and s["model"] in tab):
            if all(q.has_output(d) for d in s["after"]):
                os.replace(q.p("pending", jid), q.p("running", jid))
                import run as runmod
                assert runmod.run_job(cfg, q, s, log_to_stdout=False), open(os.path.join(RES, "logs", jid + ".log")).read()[-3000:]
    done = [json.load(open(q.out(j))) for j in os.listdir(os.path.join(RES, "runs")) for j in [j[:-5]]]
    kinds = sorted({r["spec"]["kind"] for r in done})
    print("  completed kinds:", kinds, "runs:", len(done))
    for r in done:
        if r["spec"]["kind"] == "survival":
            print("  survival", r["spec"]["dataset"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["result"].items()})
        if r["spec"]["kind"] == "tabular":
            print("  tabular", r["spec"]["model"], r["spec"]["dataset"], r["spec"]["split"],
                  "ports" if r["cfg"]["data"].get("ports", True) else "no-ports",
                  f"macro-F1 {r['result']['metrics']['macro_f1']:.3f}")
    assert "coverage" in kinds and "survival" in kinds and "sizes" in kinds and "tabular" in kinds


if __name__ == "__main__":
    cfg = cfg_for_tests()
    shutil.rmtree(CACHE, ignore_errors=True)
    test_line_graph()
    test_sampler(cfg)
    test_pipeline(cfg)
    for script in ("make_tables.py", "make_figures.py"):
        if os.path.exists(script):
            r = subprocess.run([sys.executable, script, "--results", RES], capture_output=True, text=True)
            print(r.stdout[-1500:], r.stderr[-1500:])
            assert r.returncode == 0, script
    print("ALL NUMPY-SIDE TESTS PASSED")
