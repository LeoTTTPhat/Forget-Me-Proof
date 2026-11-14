"""Figures from results/<run>/analysis (and summary.json). PDFs go to figures/ (or figures/<run>/ for non-main runs).
Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red), thin marks,
recessive grid, single y-axis per panel, legends + direct labels where <= 4 series.

python src/plots.py [--run main]
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import ROOT, RESULTS, load_json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.5})
ORDER = ["base", "retrain", "planted", "ga", "graddiff", "npo", "rmu", "adaptive50", "adaptive90",
         "filter_kw_full", "filter_kw_tok", "filter_semantic", "route_semantic"]


def lj(p):
    return load_json(p) if os.path.exists(p) else None


def fig_power(main, out, alpha="0.05"):
    tags = sorted({k.split("/")[0] for k in main})
    for tag in tags:
        vs = [v for v in ORDER if f"{tag}/{v}" in main] + sorted(v.split("/")[1] for v in main if v.startswith(tag + "/") and v.split("/")[1] not in ORDER)
        if not vs:
            continue
        x = np.arange(len(vs))
        fig, ax = plt.subplots(figsize=(6.8, 2.4))
        for i, (role, lab) in enumerate([("deleted", "deleted subjects (power / Type-I if compliant)"), ("phantom", "phantom subjects (exact null)")]):
            vals, lo, hi = [], [], []
            for v in vs:
                d = main[f"{tag}/{v}"].get(role, {}).get(f"alpha={alpha}")
                r = d["reject_rate"] if d else np.nan
                ci = d["ci95"] if d else [np.nan, np.nan]
                vals.append(r); lo.append(r - ci[0]); hi.append(ci[1] - r)
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=SERIES[i], label=lab, yerr=[lo, hi], ecolor=INK2, capsize=1.5,
                   error_kw={"elinewidth": 0.7})
        ax.axhline(float(alpha), color=SERIES[7], lw=0.8, ls="--")
        ax.text(-0.45, float(alpha) + 0.02, f"alpha = {alpha}", color=INK2, ha="left", fontsize=7)
        ax.set_xticks(x, vs, rotation=30, ha="right")
        ax.set_ylabel("rejection rate"); ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"power_{tag}.pdf")); plt.close(fig)


def fig_ablation(abl, out, key_prefix, fname, xlabel, alpha="0.05"):
    variants = [v for v in abl if any(k.startswith(key_prefix) for k in abl[v])]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, v in enumerate(variants[:4]):
        ks = [k for k in abl[v] if k.startswith(key_prefix)]
        xs, ys = [], []
        for k in ks:
            d = abl[v][k].get("deleted", {}).get(f"alpha={alpha}")
            if d:
                xs.append(k.split("=", 1)[1]); ys.append(d["reject_rate"])
        ax.plot(range(len(xs)), ys, marker="o", ms=4, color=SERIES[i], label=v)
        ax.set_xticks(range(len(xs)), xs, rotation=20)
    ax.set_xlabel(xlabel); ax.set_ylabel("power (deleted subjects)"); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, fname)); plt.close(fig)


def fig_typeI(t1, mc, out):
    if not t1:
        return
    items = [(k, v) for k, v in t1.items()]
    if mc:
        items += [(f"MC {k}", v) for k, v in mc.items() if "plugin" in k or "peek" in k][:8]
    fig, ax = plt.subplots(figsize=(6.8, 0.25 * len(items) + 0.8))
    for i, (k, v) in enumerate(items):
        col = SERIES[7] if "peek" in k else SERIES[0]
        ax.plot([v["ci95"][0], v["ci95"][1]], [i, i], color=col, lw=1.2)
        ax.plot(v["rate"], i, "o", ms=4, color=col)
    ax.axvline(0.05, color=INK2, ls="--", lw=0.8); ax.axvline(0.01, color=INK2, ls=":", lw=0.8)
    ax.set_yticks(range(len(items)), [f"{k} (n={v['n']})" for k, v in items], fontsize=6)
    ax.set_xlabel("false-accusation rate (95% Clopper-Pearson)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "typeI.pdf")); plt.close(fig)


def fig_detector(det, out):
    seeds = {k: v for k, v in (det or {}).items() if k.startswith("detector_")}
    if not seeds:
        return
    from fmp import eprocess as E
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    s = np.linspace(0, 1, 101)
    for i, (k, v) in enumerate(sorted(seeds.items())):
        auc = v["target_hull_auc"]
        ax.plot(s, [E.prop4_bound(x, auc) for x in s], color=SERIES[i], lw=1, label=f"{k}: bound (hull AUC {auc:.2f})")
        for op in v["operating_points"]:
            ax.plot(op["tpr"], op["fpr_genuine"], "o", ms=5, color=SERIES[i])
    ax.set_xlabel("canary suppression rate s (TPR)"); ax.set_ylabel("genuine content suppressed (FPR)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "adaptive_cost.pdf")); plt.close(fig)


def fig_rag(rag, out, alpha="0.05"):
    if not rag:
        return
    behs = [b for b in ["compliant", "lazy", "partial", "filter_kw_full", "filter_kw_tok", "filter_semantic"] if b in rag]
    vals = [rag[b].get("deleted", {}).get(f"alpha={alpha}", {}).get("reject_rate", np.nan) for b in behs]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(range(len(behs)), vals, 0.6, color=SERIES[0])
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(float(alpha), color=SERIES[7], ls="--", lw=0.8)
    ax.set_xticks(range(len(behs)), behs, rotation=30, ha="right"); ax.set_ylim(0, 1.1); ax.set_ylabel("rejection rate (deleted)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "rag.pdf")); plt.close(fig)


def fig_ranklaw(main, out):
    tags = sorted({k.split("/")[0] for k in main})
    for tag in tags:
        vs = [v for v in ["retrain", "planted", "npo", "ga", "rmu"] if f"{tag}/{v}" in main and main[f"{tag}/{v}"]["deleted"].get("rank_law")]
        if not vs:
            continue
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for i, v in enumerate(vs[:5]):
            h = np.array(main[f"{tag}/{v}"]["deleted"]["rank_law"]["hist"]); h = h / h.sum()
            ax.plot(np.arange(1, len(h) + 1), h, marker="o", ms=3, color=SERIES[i], label=v)
        ax.axhline(0.1, color=INK2, ls="--", lw=0.8)
        ax.set_xlabel("rank of planted candidate (1 = top)"); ax.set_ylabel("fraction of canaries"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"ranklaw_{tag}.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    an = os.path.join(RESULTS, a.run, "analysis")
    out = os.path.join(ROOT, "figures") if a.run == "main" else os.path.join(ROOT, "figures", a.run)
    os.makedirs(out, exist_ok=True)
    main_ = lj(os.path.join(an, "main.json")) or {}
    abl = lj(os.path.join(an, "ablations.json")) or {}
    fig_power(main_, out)
    fig_ablation(abl, out, "M=", "ablation_M.pdf", "candidates per canary (m+1)")
    fig_ablation(abl, out, "n_canaries=", "ablation_ncanaries.pdf", "canaries per subject")
    fig_ablation(abl, out, "strategy=", "ablation_strategy.pdf", "betting strategy")
    fig_ablation(abl, out, "kinds=", "ablation_kinds.pdf", "probe types")
    fig_typeI(lj(os.path.join(an, "typeI_empirical.json")), lj(os.path.join(an, "mc_null.json")), out)
    fig_detector(lj(os.path.join(an, "detector.json")), out)
    fig_rag(lj(os.path.join(an, "rag.json")), out)
    fig_ranklaw(main_, out)
    print("figures ->", out, sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
