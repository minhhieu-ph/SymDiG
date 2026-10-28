"""All result tables (Markdown summary + LaTeX tabular bodies) from results/runs/.

  python make_tables.py [--results results]
  -> results/tables/summary.md and results/tables/E*.tex
Numbers are mean ± std over seeds (macro-F1 in %, full test set unless stated).
"""
import argparse
import os

import numpy as np

from symdig.report import (DS_ORDER, DS_SHORT, MODEL_NAMES, RUNG_NAMES, load_runs, macro, mstat, paired_test,
                           select)


class Out:
    def __init__(self, d):
        self.d, self.md = d, ["# SymDiG results (auto-generated)\n"]

    def table(self, name, title, header, rows, note=None):
        if not rows:
            return
        self.md.append(f"\n## {title}\n")
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "---|" * len(header))
        for r in rows:
            self.md.append("| " + " | ".join(str(x) for x in r) + " |")
        if note:
            self.md.append(f"\n{note}\n")
        tex = [" & ".join(header) + r" \\", r"\midrule"]
        tex += [" & ".join(str(x).replace("±", r"$\pm$").replace("%", r"\%") for x in r) + r" \\" for r in rows]
        with open(os.path.join(self.d, f"{name}.tex"), "w") as fh:
            fh.write("\n".join(tex) + "\n")

    def save(self):
        with open(os.path.join(self.d, "summary.md"), "w") as fh:
            fh.write("\n".join(self.md) + "\n")


def present_ds(runs):
    s = {r["spec"]["dataset"] for r in runs}
    return [d for d in DS_ORDER if d in s]


def e1(R, O):
    rows = []
    for r in sorted(select(R, kind="coverage"), key=lambda r: (DS_ORDER.index(r["spec"]["dataset"])
                                                               if r["spec"]["dataset"] in DS_ORDER else 9,
                                                               r["spec"]["fanout"][0])):
        x = r["result"]
        rows.append([DS_SHORT.get(r["spec"]["dataset"], r["spec"]["dataset"]), str(r["spec"]["fanout"]),
                     f"{100 * x['multignn']['seed_coverage']:.1f}", f"{100 * x['multignn']['twin_coverage']:.1f}",
                     f"{100 * x['asym_forced']['twin_coverage']:.1f}", f"{x['multignn']['edges_per_batch']:.0f}",
                     f"{x['coupled_forced']['edges_per_batch']:.0f}"])
    O.table("E1_coverage", "E1 — Sampler diagnosis on the training graph (no training)",
            ["Dataset", "Fan-out", "Seed coverage, Multi-GNN (%)", "Twin coverage, Multi-GNN (%)",
             "Twin coverage, forced (%)", "Edges/batch Multi-GNN", "Edges/batch coupled"], rows,
            "Seed coverage = share of training targets that enter the computation graph and receive a loss. "
            "Twin coverage = share of sampled forward edges whose reverse twin is also sampled.")
    rows = []
    for d in present_ds(select(R, kind="gnn", preset="multignn_paper")):
        rs = select(R, kind="gnn", preset="multignn_paper", dataset=d)
        rows.append([DS_SHORT.get(d, d), mstat(macro(rs, "metrics_covered"))[2], mstat(macro(rs))[2],
                     mstat([r["result"].get("train_seed_coverage") for r in rs])[2],
                     mstat([r["result"].get("test_seed_coverage") for r in rs])[2], len(rs)])
    O.table("E1_multignn", "E1 — Multi-GNN with its released hyper-parameters (fan-out 100)",
            ["Dataset", "Macro-F1 on sampled test targets (as the original code reports)",
             "Macro-F1 on 100% of test targets", "Train seed coverage (%)", "Test seed coverage (%)", "Seeds"], rows)


def e2(R, O, exp_datasets=None):
    G = select(R, kind="gnn", split="p1A", task="multi")
    lad = [r for r in G if str(r["spec"].get("model", "")).startswith("ladder") and not r["spec"].get("preset")
           and not r["spec"]["dataset"].startswith("motif")]
    dss = present_ds(lad)
    if not lad:
        return
    rows, by = [], {}
    for k in range(5):
        row = [RUNG_NAMES[k]]
        for d in dss:
            rs = [r for r in lad if r["spec"]["dataset"] == d and r["spec"]["model"] == f"ladder{k}"]
            by[(k, d)] = rs
            row.append(mstat(macro(rs))[2])
        rows.append(row)
    for k in range(1, 5):
        row = [f"Δ ({k}) vs ({k - 1}) [p]"]
        for d in dss:
            m, p, n = paired_test(by.get((k - 1, d), []), by.get((k, d), []))
            row.append("--" if m is None else f"{m:+.2f} [{p:.3f}]")
        rows.append(row)
    rows.append(["(0) on sampled targets only"] + [mstat(macro(by.get((0, d), []), "metrics_covered"))[2] for d in dss])
    rows.append(["(0) test seed coverage (%)"] + [mstat([r["result"].get("test_seed_coverage") for r in by.get((0, d), [])])[2]
                                                  for d in dss])
    O.table("E2_ladder", "E2 — Ablation ladder (macro-F1 %, full test set, mean ± std over seeds)",
            ["Rung"] + [DS_SHORT[d] for d in dss], rows,
            "Each rung changes one factor with respect to the previous one; same backbone and hyper-parameters. "
            "p = two-sided sign-flip permutation test over paired seeds (minimum attainable p = 2/2^n).")
    # per-class, ToN-like
    for d in dss:
        r2, r4 = by.get((2, d), []), by.get((4, d), [])
        if not r2 or not r4:
            continue
        classes = r4[0]["result"]["classes"]
        rows = []
        for c in classes:
            sup = r4[0]["result"]["metrics"]["per_class"][c]["support"]
            f = lambda rs: mstat([r["result"]["metrics"]["per_class"][c]["f1"] for r in rs])[2]
            rows.append([c, f"{sup:,}", f(by.get((0, d), [])), f(r2), f(r4)])
        O.table(f"E2_perclass_{d}", f"E2 — Per-class F1 on {DS_SHORT[d]} (rungs 0, 2, 4)",
                ["Class", "Test support", "(0) Multi-GNN pipeline", "(2) coupled, per-layer mean", "(4) SymDiG"], rows)


def e3(R, O):
    S = sorted(select(R, kind="survival"), key=lambda r: r["result"].get("multiplicity") or 0)
    rows = [[r["result"]["multiplicity"], f"{r['result']['asym_strict']:.3f} / {r['result']['pred_asym_strict']:.3f}",
             f"{r['result']['asym_fwd']:.3f} / {r['result']['pred_asym_fwd']:.3f}",
             f"{r['result']['coupled']:.3f} / {r['result']['pred_coupled']:.3f}", r["result"]["n"]] for r in S]
    O.table("E3_survival", "E3 — Probability that a planted cycle survives sampling (empirical / predicted)",
            ["Multiplicity M", "Asymmetric, both relations", "Asymmetric, forward only", "Coupled", "Instances"], rows)
    G = [r for r in select(R, kind="gnn") if r["spec"]["dataset"].startswith("motif")]
    Ms = sorted({int(r["spec"]["dataset"].split("_m")[1]) for r in G})
    if not G:
        return
    rows = []
    for k in range(5):
        row = [RUNG_NAMES[k]]
        for M in Ms:
            rs = [r for r in G if r["spec"]["dataset"] == f"motif_m{M}" and r["spec"]["model"] == f"ladder{k}"]
            row.append(mstat(macro(rs))[2])
        rows.append(row)
    O.table("E3_ladder", "E3 — Macro-F1 (%) on structure-only motif labels vs multiplicity",
            ["Rung"] + [f"M={M}" for M in Ms], rows)
    classes = ["FanIn", "FanOut", "Cycle", "ScatterGather"]
    rows = []
    for M in Ms:
        for k in (2, 3, 4):
            rs = [r for r in G if r["spec"]["dataset"] == f"motif_m{M}" and r["spec"]["model"] == f"ladder{k}"]
            if rs:
                rows.append([M, RUNG_NAMES[k]] + [mstat([r["result"]["metrics"]["per_class"][c]["f1"] for r in rs])[2]
                                                  for c in classes])
    O.table("E3_perclass", "E3 — Per-motif F1 (%): does late fusion lose mixed-direction motifs?",
            ["M", "Rung"] + classes, rows)


def model_rows(R, split, models, dss, ports=True):
    rows = []
    for m in models:
        row = [MODEL_NAMES.get(m, m)]
        any_ = False
        for d in dss:
            rs = select(R, split=split, model=m, dataset=d, task="multi")
            rs = [r for r in rs if r["spec"]["kind"] in ("gnn", "tabular", "fecograph") and
                  (r["spec"]["kind"] == "fecograph" or (r.get("cfg") or {}).get("data", {}).get("ports", True) == ports)]
            any_ |= bool(rs)
            row.append(mstat(macro(rs))[2])
        if any_:
            rows.append(row)
    return rows


def e4(R, O, cfg_models):
    dss = present_ds(select(R, split="p1A"))
    O.table("E4_main", "E4 — All models in one frame (split p1A, tuned, macro-F1 %)",
            ["Model"] + [DS_SHORT[d] for d in dss], model_rows(R, "p1A", cfg_models, dss))
    O.table("E4_noports", "E4 — Without port features",
            ["Model"] + [DS_SHORT[d] for d in dss], model_rows(R, "p1A", ["xgboost", "symdig"], dss, ports=False))
    d2 = present_ds(select(R, split="p2"))
    O.table("E4_p2", "E4 — FeCoGraph protocol p2 (2% subsample, 30% labeled), same split for all",
            ["Model"] + [DS_SHORT[d] for d in d2],
            model_rows(R, "p2", ["xgboost", "rf", "egsage", "symdig", "fecograph"], d2))
    for d in dss:
        a = select(R, split="p1A", model="symdig", dataset=d, task="multi", ports=True)
        b = select(R, split="p1A", model="xgboost", dataset=d, task="multi", ports=True)
        c0 = select(R, split="p1A", model="multignn", dataset=d, task="multi", ports=True)
        if not a or not b:
            continue
        rows = []
        for c in a[0]["result"]["classes"]:
            f = lambda rs: mstat([r["result"]["metrics"]["per_class"][c]["f1"] for r in rs])[2] if rs else "--"
            rows.append([c, f"{a[0]['result']['metrics']['per_class'][c]['support']:,}", f(b), f(c0), f(a)])
        O.table(f"E4_perclass_{d}", f"E4 — Per-class F1 on {DS_SHORT[d]}: where does the graph help?",
                ["Class", "Test support", "XGBoost", "Multi-GNN", "SymDiG"], rows)


def e5(R, O):
    dss = present_ds(select(R, split="p1B"))
    rows = []
    for m in ("symdig", "multignn", "xgboost", "mlp"):
        for d in dss:
            a = select(R, split="p1A", model=m, dataset=d, task="multi", ports=True)
            b = select(R, split="p1B", model=m, dataset=d, task="multi", ports=True)
            b = [r for r in b if r["spec"]["kind"] in ("gnn", "tabular")]
            if not b:
                continue
            ma, mb = mstat(macro(a)), mstat(macro(b))
            excl = b[0]["result"].get("split_meta", {}).get("excluded_classes")
            rows.append([MODEL_NAMES.get(m, m), DS_SHORT[d], ma[2], mb[2],
                         "--" if ma[0] is None or mb[0] is None else f"{mb[0] - ma[0]:+.2f}", excl if excl else "-"])
    O.table("E5_host", "E5 — Random split (p1A) vs attacker-disjoint split (p1B), macro-F1 %",
            ["Model", "Dataset", "p1A", "p1B", "Δ", "Classes excluded in p1B"], rows,
            "p1B macro-F1 is computed over the classes that could be split by attacker source IP.")
    T = select(R, kind="transfer")
    if T:
        rows = []
        for m in sorted({r["spec"]["model"] for r in T}):
            for src in DS_ORDER:
                row = [MODEL_NAMES.get(m, m), DS_SHORT[src]]
                ok = False
                for tgt in DS_ORDER:
                    rs = [r for r in T if r["spec"]["model"] == m and r["spec"]["source"] == src and r["spec"]["dataset"] == tgt]
                    ok |= bool(rs)
                    row.append("—" if src == tgt else mstat(macro(rs))[2])
                if ok:
                    rows.append(row)
        O.table("E5_transfer", "E5 — Cross-dataset binary transfer (train on row, test on column), macro-F1 %",
                ["Model", "Train on"] + [DS_SHORT[d] for d in DS_ORDER], rows)


def e6(R, O):
    rows = []
    for r in select(R, kind="sizes"):
        for f, x in sorted(r["result"].items(), key=lambda kv: float(kv[0])):
            rows.append([DS_SHORT.get(r["spec"]["dataset"], r["spec"]["dataset"]), f"{float(f) * 100:g}%", f"{x['flows']:,}",
                         f"{x['ip_nodes']:,}", f"{x['ipport_nodes']:,}", f"{x['line_graph_edges_ipport']:.3g}"])
    O.table("E6_sizes", "E6 — Graph sizes on the same flows",
            ["Dataset", "Share of flows", "Flows (= edges)", "Nodes, IP-centric", "Nodes, IP:PORT",
             "Line-graph edges (IP:PORT hosts)"], rows)
    rows = []
    for r in sorted(select(R, kind="deploy"), key=lambda r: (r["spec"]["dataset"], r["spec"]["model"],
                                                             r["spec"].get("frac", 1), str(r["spec"].get("mem_cap_gb")))):
        x = r["result"]
        if x.get("status") != "ok":
            rows.append([DS_SHORT.get(r["spec"]["dataset"]), MODEL_NAMES.get(r["spec"]["model"]), x.get("frac"),
                         x.get("mem_cap_gb") or "40", x.get("status"), "", "", "", ""])
            continue
        tr = x["train"]
        inf = x.get(f"infer_bs{8192}") or next((v for k, v in x.items() if k.startswith("infer_bs")), {})
        rows.append([DS_SHORT.get(r["spec"]["dataset"]), MODEL_NAMES.get(r["spec"]["model"]), x.get("frac"),
                     x.get("mem_cap_gb") or "40", "ok", f"{tr.get('peak_vram_gb') or 0:.2f}",
                     f"{tr.get('targets_per_sec') or 0:,.0f}", f"{inf.get('flows_per_sec_end_to_end', 0):,.0f}",
                     f"{inf.get('gpu_ms_p95', 0):.1f}"])
    O.table("E6_deploy", "E6 — Training and inference cost on one A100",
            ["Dataset", "Model", "Share of flows", "Memory cap (GB)", "Status", "Train peak VRAM (GB)",
             "Train targets/s", "Inference flows/s (end-to-end)", "Inference p95 GPU ms/batch"], rows)


def e7(R, O):
    E = select(R, kind="evasion")
    if not E:
        return
    rows = []
    for m in sorted({r["spec"]["model"] for r in E}):
        rs = [r for r in E if r["spec"]["model"] == m]
        strats = rs[0]["result"]["results"].keys()
        for s in strats:
            ks = sorted(rs[0]["result"]["results"][s].keys(), key=int)
            rows.append([MODEL_NAMES.get(m, m), s] + [mstat([r["result"]["results"][s][k]["mean_f1"] for r in rs])[2] for k in ks])
    ks = sorted(E[0]["result"]["results"][next(iter(E[0]["result"]["results"]))].keys(), key=int)
    O.table("E7_evasion", f"E7 — Mean F1 (%) of the attacked classes {E[0]['result']['attacked']} under injection",
            ["Model", "Strategy"] + [f"K={k}" for k in ks], rows,
            "XGBoost and the MLP score each flow independently, so injected chaff cannot change their predictions "
            "on the attack flows; compare with their clean per-class F1 in E4.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = load_runs(a.results)
    d = os.path.join(a.results, "tables")
    os.makedirs(d, exist_ok=True)
    O = Out(d)
    models = ["xgboost", "rf", "mlp", "egsage", "dirgnn", "multignn", "multignn_fixed", "symdig"]
    for f in (e1, e2, e3, lambda R, O: e4(R, O, models), e5, e6, e7):
        try:
            f(R, O)
        except Exception as ex:  # partial results are normal while experiments are running
            O.md.append(f"\n(skipped a table: {type(ex).__name__}: {ex})\n")
    O.save()
    print("\n".join(O.md))
    print(f"\nwrote {d}/summary.md and {d}/*.tex")


if __name__ == "__main__":
    main()
