"""Figures for the paper from results/runs/ (vector PDF + PNG in results/figures/).

  fig_e1_coverage    seed coverage of the Multi-GNN sampler vs node degree (E1)
  fig_e2_ladder      macro-F1 per rung of the ablation ladder (E2)
  fig_e3_survival    motif survival vs multiplicity: empirical vs Proposition 1 (E3)
  fig_e3_ladder      macro-F1 on structure-only labels vs multiplicity (E3)
  fig_e4_perclass    per-class F1, XGBoost vs Multi-GNN vs SymDiG (E4)
  fig_e6_cost        VRAM and throughput, IP-centric vs IP:PORT models (E6)
  fig_e7_evasion     attacked-class F1 vs injected chaff per strategy (E7)
"""
import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

from symdig.report import DS_ORDER, DS_SHORT, MODEL_NAMES, RUNG_NAMES, load_runs, macro, select  # noqa: E402

C = {"symdig": "#1f5fa8", "multignn": "#c4600f", "xgboost": "#4b8a3e", "egsage": "#8a8a8a", "multignn_fixed": "#e0a060",
     "dirgnn": "#7a5aa6", "rf": "#9cc58f", "mlp": "#b7b7b7"}
RUNG_C = ["#c4600f", "#e0a060", "#9a9a9a", "#7fa7d6", "#1f5fa8"]


def style():
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    for f in ("regular", "bold", "italic", "bolditalic"):
        p = f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{f}.otf"
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    plt.rcParams.update({"font.family": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def ms(vals):
    v = np.array([x for x in vals if x is not None], float) * 100
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0) if len(v) else (np.nan, 0.0)


def fig_e1(R, out):
    rs = [r for r in select(R, kind="coverage") if r["result"]["multignn"]["by_degree"]]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for r in sorted(rs, key=lambda r: (r["spec"]["dataset"], r["spec"]["fanout"][0])):
        bd = r["result"]["multignn"]["by_degree"]
        x = [np.sqrt(max(b["lo"], 1) * (b["hi"] or b["lo"] * 10)) for b in bd]
        y = [100 * b["coverage"] for b in bd]
        ls = {20: ":", 50: "--", 100: "-"}.get(r["spec"]["fanout"][0], "-")
        ax.plot(x, y, ls, marker="o", ms=2.5, lw=1,
                label=f"{DS_SHORT.get(r['spec']['dataset'], r['spec']['dataset'])}, B={r['spec']['fanout'][0]}")
    ax.set_xscale("log")
    ax.set_xlabel("multigraph degree of the target's endpoints")
    ax.set_ylabel("targets that receive a loss (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    ax.grid(lw=0.3, color="#e5e5e5")
    save(fig, out, "fig_e1_coverage")


def fig_e2(R, out):
    lad = [r for r in select(R, kind="gnn", split="p1A") if str(r["spec"].get("model", "")).startswith("ladder")
           and not r["spec"].get("preset") and not r["spec"]["dataset"].startswith("motif")]
    dss = [d for d in DS_ORDER if any(r["spec"]["dataset"] == d for r in lad)]
    if not dss:
        return
    fig, ax = plt.subplots(figsize=(7.16, 1.9))
    w = 0.8 / 5
    x = np.arange(len(dss))
    for k in range(5):
        st = [ms(macro([r for r in lad if r["spec"]["dataset"] == d and r["spec"]["model"] == f"ladder{k}"])) for d in dss]
        ax.bar(x + (k - 2) * w, [s[0] for s in st], w, yerr=[s[1] for s in st], color=RUNG_C[k], label=RUNG_NAMES[k],
               lw=0, error_kw={"lw": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([DS_SHORT[d] for d in dss])
    ax.set_ylabel("macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.grid(axis="y", lw=0.3, color="#e5e5e5")
    ax.set_axisbelow(True)
    save(fig, out, "fig_e2_ladder")


def fig_e3(R, out):
    S = sorted(select(R, kind="survival"), key=lambda r: r["result"]["multiplicity"])
    if S:
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        M = [r["result"]["multiplicity"] for r in S]
        for key, lab, col in [("asym_strict", "asymmetric (both relations)", "#c4600f"),
                              ("asym_fwd", "asymmetric (forward only)", "#e0a060"), ("coupled", "coupled", "#1f5fa8")]:
            ax.plot(M, [r["result"][key] for r in S], "o", color=col, ms=3, label=f"{lab}, measured")
            ax.plot(M, [r["result"]["pred_" + key] for r in S], "-", color=col, lw=1, alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("edge multiplicity M")
        ax.set_ylabel("P(planted cycle survives)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=5.5)
        ax.text(0.98, 0.02, "lines: Proposition 1", transform=ax.transAxes, ha="right", fontsize=6, color="#666")
        ax.grid(lw=0.3, color="#e5e5e5")
        save(fig, out, "fig_e3_survival")
    G = [r for r in select(R, kind="gnn") if r["spec"]["dataset"].startswith("motif")]
    if G:
        Ms = sorted({int(r["spec"]["dataset"].split("_m")[1]) for r in G})
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        for k in range(5):
            st = [ms(macro([r for r in G if r["spec"]["dataset"] == f"motif_m{m}" and r["spec"]["model"] == f"ladder{k}"]))
                  for m in Ms]
            ax.errorbar(Ms, [s[0] for s in st], [s[1] for s in st], color=RUNG_C[k], lw=1, ms=3, marker="o",
                        label=RUNG_NAMES[k], capsize=1.5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("edge multiplicity M")
        ax.set_ylabel("macro-F1 (%)")
        ax.legend(frameon=False, fontsize=5.5)
        ax.grid(lw=0.3, color="#e5e5e5")
        save(fig, out, "fig_e3_ladder")


def fig_e4(R, out):
    dss = [d for d in DS_ORDER if select(R, split="p1A", model="symdig", dataset=d, task="multi", ports=True)]
    if not dss:
        return
    fig, axes = plt.subplots(len(dss), 1, figsize=(7.16, 1.5 * len(dss)), squeeze=False)
    for ax, d in zip(axes[:, 0], dss):
        series = [(m, select(R, split="p1A", model=m, dataset=d, task="multi", ports=True)) for m in ("xgboost", "multignn", "symdig")]
        series = [(m, rs) for m, rs in series if rs]
        classes = series[-1][1][0]["result"]["classes"]
        x = np.arange(len(classes))
        w = 0.8 / len(series)
        for j, (m, rs) in enumerate(series):
            st = [ms([r["result"]["metrics"]["per_class"][c]["f1"] for r in rs]) for c in classes]
            ax.bar(x + (j - (len(series) - 1) / 2) * w, [s[0] for s in st], w, yerr=[s[1] for s in st], color=C[m],
                   label=MODEL_NAMES[m], lw=0, error_kw={"lw": 0.5})
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 105)
        ax.set_ylabel("F1 (%)")
        ax.set_title(DS_SHORT[d], loc="left", fontsize=7.5)
        ax.grid(axis="y", lw=0.3, color="#e5e5e5")
        ax.set_axisbelow(True)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "fig_e4_perclass")


def fig_e6(R, out):
    D = [r for r in select(R, kind="deploy") if r["result"].get("status") == "ok" and r["spec"].get("frac", 1) == 1.0
         and not r["spec"].get("mem_cap_gb")]
    if not D:
        return
    dss = [d for d in DS_ORDER if any(r["spec"]["dataset"] == d for r in D)]
    models = [m for m in ("egsage", "multignn", "symdig") if any(r["spec"]["model"] == m for r in D)]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.8))
    w = 0.8 / max(len(models), 1)
    for ax, (key, lab) in zip(axes, [("vram", "training peak VRAM (GB)"), ("thr", "inference flows / s")]):
        for j, m in enumerate(models):
            vals = []
            for d in dss:
                rs = [r for r in D if r["spec"]["dataset"] == d and r["spec"]["model"] == m]
                if not rs:
                    vals.append(np.nan)
                    continue
                x = rs[0]["result"]
                if key == "vram":
                    vals.append(x["train"].get("peak_vram_gb") or np.nan)
                else:
                    inf = next((v for k, v in x.items() if k.startswith("infer_bs") and k != "infer_bs1024"), {})
                    vals.append(inf.get("flows_per_sec_end_to_end", np.nan))
            ax.bar(np.arange(len(dss)) + (j - (len(models) - 1) / 2) * w, vals, w, color=C[m], label=MODEL_NAMES[m], lw=0)
        ax.set_xticks(np.arange(len(dss)))
        ax.set_xticklabels([DS_SHORT[d] for d in dss])
        ax.set_ylabel(lab)
        ax.grid(axis="y", lw=0.3, color="#e5e5e5")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig_e6_cost")


def fig_e7(R, out):
    E = select(R, kind="evasion")
    if not E:
        return
    strats = list(E[0]["result"]["results"].keys())
    fig, axes = plt.subplots(1, len(strats), figsize=(7.16, 1.8), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], strats):
        for m in sorted({r["spec"]["model"] for r in E}):
            rs = [r for r in E if r["spec"]["model"] == m]
            ks = sorted(rs[0]["result"]["results"][s].keys(), key=int)
            st = [ms([r["result"]["results"][s][k]["mean_f1"] for r in rs]) for k in ks]
            ax.errorbar([int(k) for k in ks], [v[0] for v in st], [v[1] for v in st], marker="o", ms=3, lw=1,
                        color=C.get(m, "#444"), label=MODEL_NAMES.get(m, m), capsize=1.5)
        ax.set_title(f"strategy: {s}", fontsize=7.5)
        ax.set_xlabel("chaff flows per attack flow (K)")
        ax.grid(lw=0.3, color="#e5e5e5")
    axes[0, 0].set_ylabel("attacked-class F1 (%)")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig_e7_evasion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    style()
    R = load_runs(a.results)
    out = os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    for f in (fig_e1, fig_e2, fig_e3, fig_e4, fig_e6, fig_e7):
        try:
            f(R, out)
        except Exception as ex:
            print(f"  skipped {f.__name__}: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
