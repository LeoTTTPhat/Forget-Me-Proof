"""Stage 5: paper figures (vector PDF) from results/ only. CPU, seconds.

python src/make_figures.py            -> figures/fig_*.pdf  (+ figures/figures_manifest.json: figure -> claim -> source files)

Inputs: results/summary.json, results/stats_tests.json, results/calibration_mc.json, results/<run>/analysis/*.json.
Style: IEEE two-column (single column 3.5 in, double 7.16 in), Times-like serif at 7-8 pt, validated colour-blind-safe
categorical palette (dataviz reference palette; adjacent-pair CVD dE >= 9.1) used in a FIXED behaviour -> colour mapping,
always with a secondary encoding (marker shape and/or direct labels), recessive grid, one y-axis per panel.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
W1, W2 = 3.5, 7.16

INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# fixed behaviour -> (colour, marker, label); compliant/null are neutral inks
BEH = {"planted": (PAL[0], "o", "Lazy (no deletion)"), "npo": (PAL[1], "s", "NPO"), "rmu": (PAL[2], "^", "RMU"),
       "graddiff": (PAL[3], "D", "GradDiff"), "adaptive50": (PAL[4], "v", "Adaptive (TPR 0.5)"), "ga": (PAL[5], "P", "GA"),
       "adaptive90": (PAL[6], "X", "Adaptive (TPR 0.9)"), "retrain": (INK2, "o", "Compliant retrain"),
       "base": (INK3, "h", "Never trained"), "phantom (all Qwen-3B)": (INK3, ".", "Phantom subjects (null)")}
MODELS = {"qwen3b": (INK, "o", "Qwen2.5-3B"), "llama3b": (INK2, "s", "Llama-3.2-3B"), "qwen7b": (INK3, "^", "Qwen2.5-7B")}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlelocation": "left", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02})

MANIFEST = {}


def lj(*p):
    with open(os.path.join(RES, *p)) as f:
        return json.load(f)


def save(fig, name, claim, sources):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    MANIFEST[name] = dict(claim=claim, sources=sources)


def alpha_line(ax, a=0.05, horizontal=True, label=True):
    (ax.axhline if horizontal else ax.axvline)(a, color=INK, lw=0.7, ls="--", zorder=1)
    if label and horizontal:
        ax.text(ax.get_xlim()[1], a, r" $\alpha$", va="center", ha="left", fontsize=7, color=INK)


def err(v, ci):
    return [[max(0, v - ci[0])], [max(0, ci[1] - v)]]


S = lj("summary.json")
ST = lj("stats_tests.json")
MC = lj("calibration_mc.json")
AN = {r: {k: lj(r, "analysis", f"{k}.json") for k in ["main", "ablations", "ablation_m20", "memorisation", "detector",
                                                       "rag", "baselines_mia_trap", "utility"]} for r in ["main", "key1", "key2"]}


# ------------------------------------------------------------------------------------------------ Fig. 1 calibration
def fig_calibration():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.25), gridspec_kw=dict(width_ratios=[1, 1.15]))
    M = AN["main"]["main"]
    srcs = []
    base = np.array(M["qwen3b/base"]["deleted"]["rank_law"]["hist"]) + np.array(M["qwen3b/base"]["retained"]["rank_law"]["hist"])
    ret = sum(np.array(AN[r]["main"]["qwen3b/retrain"]["deleted"]["rank_law"]["hist"]) for r in ["main", "key1", "key2"])
    ph = np.zeros(10)
    for r in ["main", "key1", "key2"]:
        for vid, e in AN[r]["main"].items():
            if not vid.split("/")[1].startswith(("filter_", "route_")) and e["phantom"].get("rank_law"):
                ph += np.array(e["phantom"]["rank_law"]["hist"])
    series = [("Never-trained base (deleted + retained, n=%d)" % base.sum(), base, INK3, "h"),
              ("Compliant retrain, deleted (3 keys, n=%d)" % ret.sum(), ret, INK2, "o"),
              ("Phantoms on every trained model (n=%d)" % ph.sum(), ph, PAL[0], "s")]
    x = np.arange(1, 11)
    for i, (lab, h, c, m) in enumerate(series):
        f = h / h.sum()
        a.plot(x + (i - 1) * 0.18, f, ls="none", marker=m, color=c, ms=4, label=lab, zorder=3)
        a.vlines(x + (i - 1) * 0.18, 0, f, color=c, lw=0.8, zorder=2)
    n = ret.sum()
    lo, hi = 0.1 - 1.96 * math.sqrt(0.09 / n), 0.1 + 1.96 * math.sqrt(0.09 / n)
    a.axhspan(lo, hi, color=GRID, zorder=0)
    a.axhline(0.1, color=INK, lw=0.7, ls="--")
    a.set_xticks(x); a.set_xlabel("rank of the planted candidate among M = 10"); a.set_ylabel("fraction of null canaries")
    a.set_ylim(0, 0.2); a.legend(loc="upper left", fontsize=6, handlelength=1)
    a.set_title("(a) Null rank law is uniform (band: 95%% for n=%d)" % n)
    # (b) P(sup W >= 1/alpha) vs alpha
    C = ST["calibration_empirical"]
    g = np.array(C["alpha_grid"])
    b.plot(g, g, color=INK, lw=0.7, ls="--", label=r"Ville bound $P \leq \alpha$")
    e = C["all_orderings"]
    lo = [c[0] for c in e["ci95"]]; hi = [c[1] for c in e["ci95"]]
    b.fill_between(g, np.maximum(lo, 1e-4), hi, color=PAL[0], alpha=0.18, lw=0)
    nz = lambda y: np.where(np.asarray(y) > 0, y, np.nan)
    b.plot(g, nz(e["rate"]), color=PAL[0], lw=1.4, label="Empirical null processes (n=%d)" % e["n"])
    e0 = C["ordering0"]
    b.plot(g, nz(e0["rate"]), color=PAL[0], lw=0.9, ls=":", label="  ordering 0 only (n=%d)" % e0["n"])
    b.plot(g, nz(MC["plugin,M=10,n=9"]), color=PAL[2], lw=1.2, label=r"MC exact null, e-process ($10^5$)")
    b.plot(g, nz(MC["binom_fixedn,M=10,n=9"]), color=PAL[3], lw=1.0, ls="-.", label="MC fixed-n binomial (n=9)")
    b.plot(g, MC["binom_peek,M=10,n=9"], color=PAL[1], lw=1.0, label="MC peeking binomial (n=9)")
    b.plot(g, MC["binom_peek,twins,n=100"], color=PAL[7], lw=1.0, ls="--", label="MC peeking binomial, twins (n=100)")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(1e-3, 0.2); b.set_ylim(1e-4, 30)
    b.set_xlabel(r"nominal level $\alpha$"); b.set_ylabel("false-accusation probability")
    b.legend(loc="upper left", fontsize=5.6, handlelength=1.6, ncol=2, columnspacing=0.8)
    b.set_title("(b) False accusations stay below the Ville bound")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_calibration.pdf", "Thm 1 / E15: exact null (uniform planted ranks) and empirical false-accusation rate <= alpha for every alpha; peeking binomial exceeds alpha",
         ["results/*/analysis/main.json (rank_law)", "results/stats_tests.json:calibration_empirical", "results/calibration_mc.json"])


# ------------------------------------------------------------------------------------------------ Fig. 2 power
def fig_power():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1.45, 1]))
    order = ["retrain", "ga", "graddiff", "adaptive90", "adaptive50", "npo", "rmu", "planted"]
    A = S["across_all_seeds"]
    hl = {}
    import csv
    for r in csv.DictReader(open(os.path.join(RES, "headline.csv"))):
        hl[(r["run"], r["variant"])] = r
    for i, v in enumerate(order):
        c, m, lab = BEH[v]
        d = A[f"qwen3b/{v}"]["deleted_reject_rate"]
        reps = A[f"qwen3b/{v}"]["per_replicate"]
        vals = [x["deleted_reject_rate"] for x in reps.values()]
        jit = np.linspace(-0.12, 0.12, len(vals))
        a.plot(i + jit, vals, ls="none", marker=m, mfc="white", mec=c, ms=3.5, mew=0.8, zorder=3)
        a.errorbar(i + 0.27, d["mean"], yerr=err(d["mean"], d["ci95"]), fmt=m, color=c, ms=5, capsize=2, lw=1.0, zorder=4)
        # phantom (null) rate of the same replicates
        ph = []
        for key in reps:
            run, vid = key.split(":")
            p = hl.get((run, vid), {}).get("phantom_reject_rate")
            if p not in (None, ""):
                ph.append(float(p))
        if ph:
            a.plot(i - 0.3, np.mean(ph), ls="none", marker="_", color=INK3, ms=7, mew=1.2, zorder=3)
    a.set_xticks(range(len(order)), [BEH[v][2].replace(" (", "\n(").replace("Compliant ", "Compliant\n") for v in order], fontsize=6.1)
    a.set_ylim(-0.03, 1.03); a.set_xlim(-0.6, len(order) - 0.4); a.set_ylabel("deleted subjects rejected")
    alpha_line(a)
    a.legend(handles=[Line2D([], [], ls="none", marker="o", mfc="white", mec=INK2, label="replicate (key/FT or unlearning seed)"),
                      Line2D([], [], ls="none", marker="o", color=INK2, label="mean, 95% t-CI over replicates"),
                      Line2D([], [], ls="none", marker="_", color=INK3, mew=1.2, ms=7, label="phantom (null) rate")],
             loc="upper left", fontsize=6)
    a.set_title("(a) Qwen2.5-3B: detection by provider behaviour (n = 3-5 replicates)")
    # (b) across models, key0, subject-cluster CI
    M = AN["main"]["main"]
    cats = ["planted", "rmu", "npo", "ga", "retrain"]
    for j, (tag, (c, m, lab)) in enumerate(MODELS.items()):
        for i, v in enumerate(cats):
            e = M.get(f"{tag}/{v}")
            if not e:
                continue
            d = e["deleted"]["alpha=0.05"]
            ci = d["subject_cluster_boot"]["reject_rate_ci95"]
            b.errorbar(i + (j - 1) * 0.22, d["reject_rate"], yerr=err(d["reject_rate"], ci), fmt=m, color=c, ms=4, capsize=1.5, lw=0.9,
                       label=lab if i == 0 else None)
    b.set_xticks(range(len(cats)), [BEH[v][2].replace(" (", "\n(").replace("Compliant ", "Compliant\n") for v in cats], fontsize=6.1)
    b.set_ylim(-0.03, 1.03); b.set_ylabel("deleted subjects rejected")
    alpha_line(b, label=False)
    b.legend(loc="upper right", fontsize=6)
    b.set_title("(b) Across models (key 0; subject-cluster 95% CI)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_power.pdf", "C4/Thm 2: lazy and RMU providers detected, NPO/adaptive partially, GA/compliant not; replicates and model replication",
         ["results/summary.json:across_all_seeds", "results/headline.csv", "results/main/analysis/main.json"])


# ------------------------------------------------------------------------------------------------ Fig. 3 detection curves
def fig_detection():
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    D = ST["detection_curves"]
    show = ["planted", "rmu", "npo", "adaptive90", "graddiff", "retrain", "phantom (all Qwen-3B)"]
    for v in show:
        d = D[v]
        c, m, lab = BEH[v]
        n = np.array(d["n"])
        ax.fill_between(n, d["ci95_lo"], d["ci95_hi"], color=c, alpha=0.13, lw=0)
        ax.plot(n, d["reject_within_n"], color=c, marker=m, ms=3.2, lw=1.1, label=f"{lab} ({d['n_processes']})")
        yl = d["reject_within_n"][-1] + (0.035 if v.startswith("phantom") else 0)
        ax.text(n[-1] + 0.15, yl, lab.split(" (")[0].replace("Phantom subjects", "Phantoms (5 canaries)"), fontsize=6, va="center", color=INK)
    alpha_line(ax, label=False)
    ax.set_xlim(0.8, 11.6); ax.set_ylim(-0.02, 1.0)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("canaries queried (16 probes, 160 scoring calls each)")
    ax.set_ylabel("fraction rejected so far")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.4, ncol=1)
    sec = ax.secondary_xaxis("top", functions=(lambda x: 16 * x, lambda y: y / 16))
    sec.set_xticks([16, 48, 80, 112, 144]); sec.set_xlabel("probes", fontsize=7); sec.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, "fig_detection.pdf", "Thm 2 / queries-to-decision: anytime detection curves with subject-cluster 95% bands; most rejections by 2-4 canaries",
         ["results/stats_tests.json:detection_curves"])


# ------------------------------------------------------------------------------------------------ Fig. 4 baselines
def fig_baselines():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.65), gridspec_kw=dict(width_ratios=[1.6, 0.9, 1.0]))
    a, b, c = axs
    BP = ST["baseline_power_by_group"]
    groups = ["planted", "rmu", "npo", "adaptive50", "graddiff", "retrain"]
    tests = [("ours", "e-process, M=10 (ours)", PAL[0], "o"), ("binom10_fixedn@0.05", "fixed-n binomial, M=10", PAL[3], "D"),
             ("twin_fixedn@0.05", "twins fixed-n (Huang)", PAL[2], "^"), ("twin_ppr@0.05", "twins PPR sequential (Huang)", PAL[6], "v"),
             ("binom10_peek@0.05", "peeking binomial (invalid)", PAL[7], "x")]
    for j, (t, lab, col, m) in enumerate(tests):
        for i, g in enumerate(groups):
            d = BP[g][t]
            a.errorbar(i + (j - 2) * 0.15, d["rate"], yerr=err(d["rate"], d["ci95"]), fmt=m, color=col, ms=3.5, capsize=1.2, lw=0.8,
                       label=lab if i == 0 else None, mfc="white" if t == "binom10_peek@0.05" else col)
    short = {"planted": "Lazy", "rmu": "RMU", "npo": "NPO", "adaptive50": "Adaptive\n(TPR 0.5)", "graddiff": "Grad-\nDiff", "retrain": "Compl.\nretrain"}
    a.set_xticks(range(len(groups)), [short[g] for g in groups], fontsize=6.2)
    a.set_ylim(-0.03, 1.03); a.set_ylabel("rejection rate (deleted subjects)")
    alpha_line(a, label=False)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6, ncol=5, handlelength=1.2, columnspacing=1.0)
    a.set_title("(a) Power, Qwen-3B (all replicates)")
    # (b) size under the exact null (MC) at alpha = 0.05
    mcn = S["runs"]["main"]["mc_null"]
    g = MC["alpha_grid"]; i5 = g.index(0.05)
    rows = [("e-process (ours), n=9", MC["plugin,M=10,n=9"][i5], PAL[0]),
            ("fixed-n binomial, n=9", MC["binom_fixedn,M=10,n=9"][i5], PAL[3]),
            ("twins PPR, n=100", mcn["twins_n=100,ppr,alpha=0.05"]["rate"], PAL[6]),
            ("peeking binomial, n=9", MC["binom_peek,M=10,n=9"][i5], PAL[7]),
            ("peeking binomial, twins n=100", MC["binom_peek,twins,n=100"][i5], PAL[7])]
    for i, (lab, v, col) in enumerate(rows):
        b.barh(i, v, 0.55, color=col, alpha=0.85 if "peeking" not in lab else 0.45, edgecolor=col)
        b.text(max(v, 0.05) + 0.006, i, f"{v:.3f}", va="center", fontsize=6.3, color=INK)
    b.axvline(0.05, color=INK, lw=0.7, ls="--")
    b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.2); b.invert_yaxis()
    b.set_xlim(0, 0.26); b.set_xlabel(r"false rejection at $\alpha$=0.05 (MC)")
    b.grid(axis="y", visible=False)
    b.set_title("(b) Size under exact null")
    # (c) MIA: phantom false accusation vs deleted detection
    mia = AN["main"]["baselines_mia_trap"]["mia"]
    vs = ["qwen3b/base", "qwen3b/retrain", "qwen3b/ga", "qwen3b/npo", "qwen3b/planted", "llama3b/retrain", "llama3b/planted"]
    for i, vid in enumerate(vs):
        d = mia[vid]
        ph, de = d["phantom"]["rej@0.05"], d["deleted"]["rej@0.05"]
        c.errorbar(i - 0.12, de["rate"], yerr=err(de["rate"], de["ci95"]), fmt="o", color=PAL[0], ms=3.5, capsize=1.2, lw=0.8,
                   label="deleted (power)" if i == 0 else None)
        c.errorbar(i + 0.12, ph["rate"], yerr=err(ph["rate"], ph["ci95"]), fmt="s", color=PAL[7], ms=3.5, capsize=1.2, lw=0.8,
                   label="phantom (false accusation)" if i == 0 else None)
    c.set_xticks(range(len(vs)), [v.replace("qwen3b/", "Q-").replace("llama3b/", "L-") for v in vs], rotation=40, ha="right", fontsize=6.2)
    c.set_ylim(-0.03, 1.03); c.set_ylabel("rejection rate (n=20 subjects)")
    alpha_line(c, label=False)
    c.legend(loc="upper left", fontsize=5.8)
    c.set_title("(c) Sequential Min-K%++ MIA")
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.93))
    save(fig, "fig_baselines.pdf", "C2/C4: e-process matches fixed-n binomial power with anytime validity; m-ary beats twins; peeking and MIA are not valid",
         ["results/stats_tests.json:baseline_power_by_group", "results/calibration_mc.json", "results/summary.json:runs.main.mc_null",
          "results/main/analysis/baselines_mia_trap.json"])


# ------------------------------------------------------------------------------------------------ Fig. 5 ablations
def fig_ablations():
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.1))
    AB = AN["main"]["ablations"]
    trio = ["planted", "rmu", "npo"]

    def pt(ax, x, d, v, off, lab=True, ls="-"):
        c, m, l = BEH[v]
        ax.errorbar(x + off, d["reject_rate"], yerr=err(d["reject_rate"], d["subject_cluster_boot"]["reject_rate_ci95"]),
                    fmt=m, color=c, ms=3.5, capsize=1.2, lw=0.8, label=l if lab else None)
    # (a) m+1 on the all-candidate subset
    ax = axs[0, 0]
    m20 = AN["main"]["ablation_m20"]["qwen3b/planted"]
    for j, (st, lab, col, mk) in enumerate([("plugin", "plug-in rank bet (default)", PAL[0], "o"), ("mixture_top1", "mixture top-1 bet", PAL[2], "^"),
                                            ("top1_full", "all-in top-1 bet", PAL[7], "x")]):
        ys = [m20[f"M={M},{st}"]["deleted"]["alpha=0.05"] if "deleted" in m20[f"M={M},{st}"] else m20[f"M={M},{st}"]["alpha=0.05"] for M in [2, 5, 10, 20]]
        r = [y["reject_rate"] for y in ys]
        ci = [y["subject_cluster_boot"]["reject_rate_ci95"] for y in ys]
        ax.fill_between(range(4), [c[0] for c in ci], [c[1] for c in ci], color=col, alpha=0.12, lw=0)
        ax.plot(range(4), r, marker=mk, color=col, label=lab, ms=3.5)
    ax.set_xticks(range(4), ["2 (twins)", "5", "10", "20"]); ax.set_xlabel("candidates per canary M")
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel("rejected (deleted)"); ax.legend(fontsize=5.6, loc="lower right")
    ax.set_title("(a) Candidates M (lazy model, subset)")
    # (b) probe kinds
    ax = axs[0, 1]
    ks = [("kinds=direct", "direct"), ("kinds=paraphrase", "paraphr."), ("kinds=cloze", "cloze"), ("kinds=fewshot", "few-shot"),
          ("kinds=direct+paraphrase+cloze", "D+P+C"), ("M=10", "all 4")]
    for j, v in enumerate(trio):
        for i, (k, _) in enumerate(ks):
            pt(ax, i, AB[f"qwen3b/{v}"][k]["deleted"]["alpha=0.05"], v, (j - 1) * 0.2, lab=(i == 0))
    ax.set_xticks(range(len(ks)), [k[1] for k in ks], fontsize=6.3); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("probe types used")
    ax.legend(fontsize=5.6, loc="lower left"); ax.set_title("(b) Probe mix")
    # (c) betting strategy
    ax = axs[0, 2]
    ss = [("strategy=plugin", "plug-in\n(default)"), ("adaptive_kinds=True", "plug-in +\nprobe wts"), ("strategy=top1_fixed", "top-1\n" + r"$\lambda$=0.5"),
          ("strategy=mixture_top1", "mixture\ntop-1"), ("strategy=top1_full", "top-1\n" + r"$\lambda$=1"), ("strategy=linear_fixed", "linear\nrank")]
    for j, v in enumerate(trio):
        for i, (k, _) in enumerate(ss):
            pt(ax, i, AB[f"qwen3b/{v}"][k]["deleted"]["alpha=0.05"], v, (j - 1) * 0.2, lab=False)
    ax.set_xticks(range(len(ss)), [s[1] for s in ss], fontsize=5.9); ax.set_ylim(-0.03, 1.03)
    ax.set_title("(c) Betting strategy")
    # (d) canaries per subject
    ax = axs[1, 0]
    ns = [("n_canaries=1", 1), ("n_canaries=2", 2), ("n_canaries=3", 3), ("n_canaries=5", 5), ("M=10", 9)]
    for j, v in enumerate(trio):
        c, m, l = BEH[v]
        ys = [AB[f"qwen3b/{v}"][k]["deleted"]["alpha=0.05"] for k, _ in ns]
        x = [n for _, n in ns]
        ci = [y["subject_cluster_boot"]["reject_rate_ci95"] for y in ys]
        ax.fill_between(x, [q[0] for q in ci], [q[1] for q in ci], color=c, alpha=0.12, lw=0)
        ax.plot(x, [y["reject_rate"] for y in ys], marker=m, color=c, ms=3.5, label=l)
    ax.set_xticks([1, 2, 3, 5, 9]); ax.set_xlabel("canaries per subject"); ax.set_ylabel("rejected (deleted)"); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=5.6, loc="upper left"); ax.set_title("(d) Canaries per subject")
    # (e) planting rate (e-process on 2 canaries of each rate)
    ax = axs[1, 1]
    for j, (tag, (c, m, lab)) in enumerate(MODELS.items()):
        br = AB[f"{tag}/planted"]["by_rate_eprocess"]
        rs = ["1", "2", "4"]
        ys = [br[r]["deleted"]["alpha=0.05"] for r in rs]
        ax.errorbar(np.arange(3) + (j - 1) * 0.15, [y["reject_rate"] for y in ys],
                    yerr=np.array([err(y["reject_rate"], y["subject_cluster_boot"]["reject_rate_ci95"]) for y in ys])[:, :, 0].T,
                    fmt=m + "-", color=c, ms=3.5, capsize=1.2, lw=0.9, label=lab + " (lazy)")
    ax.set_xticks(range(3), ["r = 1", "r = 2", "r = 4"]); ax.set_xlabel("mentions per canary (2 canaries per process)")
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=5.6, loc="upper left"); ax.set_title("(e) Planting rate")
    # (f) memorisation tertile
    ax = axs[1, 2]
    mem = AN["main"]["memorisation"]["qwen3b"]["per_variant"]
    for j, v in enumerate(["planted", "rmu", "npo", "adaptive90", "graddiff", "retrain"]):
        c, m, l = BEH[v]
        y = [mem[v][str(b)]["top1"] for b in range(3)]
        ax.plot(range(3), y, marker=m, color=c, ms=3.5, label=l)
    ax.axhline(0.1, color=INK, lw=0.7, ls="--")
    ax.set_xticks(range(3), ["low", "mid", "high"]); ax.set_xlabel("pre-deletion memorisation tertile")
    ax.set_ylabel("planted top-1 rate (n=60 / tertile)"); ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=5.4, loc="upper left")
    ax.set_title("(f) Residual evidence vs memorisation")
    for ax in axs.flat:
        if ax is not axs[1, 2]:
            alpha_line(ax, label=False)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, "fig_ablations.pdf", "E14 ablations: m+1, probe mix, betting, canaries/subject, planting rate; power conditional on pre-deletion memorisation",
         ["results/main/analysis/ablations.json", "results/main/analysis/ablation_m20.json", "results/main/analysis/memorisation.json"])


# ------------------------------------------------------------------------------------------------ Fig. 6 adaptive provider
def fig_adaptive():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(width_ratios=[1.0, 1.1, 1.1]))
    det = AN["main"]["detector"]
    rows = []
    for mode, lab in [("same", "DeBERTa, exact templates"), ("heldout", "DeBERTa, held-out phrasing")]:
        v = [det[f"detector_{mode}_seed{s}"]["target_auc"] for s in range(3)]
        mn = [min(det[f"detector_{mode}_seed{s}"]["target_auc_by_type"].values()) for s in range(3)]
        rows.append((lab, v, mn))
    pf = det["ppl_filter"]
    for i, (lab, v, mn) in enumerate(rows):
        a.plot([i] * 3, v, "o", color=PAL[0], ms=3.5, label="overall (3 seeds)" if i == 0 else None)
        a.plot([i] * 3, mn, "v", mfc="white", mec=PAL[0], ms=3.5, label="worst slot type" if i == 0 else None)
    a.plot(2, pf["auc_canary_vs_genuine"], "s", color=PAL[1], ms=4, label="perplexity filter")
    a.plot(3, pf["auc_trap_vs_genuine"], "s", color=PAL[1], ms=4)
    a.axhline(0.5, color=INK, lw=0.7, ls="--")
    a.set_xticks(range(4), ["DeBERTa\nexact tmpl.", "DeBERTa\nheld-out", "ppl filter\ncanaries", "ppl filter\ntraps"], fontsize=6.2)
    a.set_ylim(0.45, 1.03); a.set_ylabel("AUC canary vs genuine statement"); a.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    a.set_title("(a) Canaries are detectable")
    # (b)/(c): fraction of canary statements targeted vs detection / genuine-fact accuracy
    ops = det["detector_same_seed0"]["operating_points"]
    A = S["across_all_seeds"]
    pts = [("planted", 0.0), ("adaptive50", ops[0]["tpr"]), ("adaptive90", ops[1]["tpr"]), ("npo", 1.0)]
    for v, x in pts:
        col, m, lab = BEH[v]
        reps = A[f"qwen3b/{v}"]["per_replicate"]
        if v == "npo":  # same planted model (key0) only, so x and the flagged set are comparable
            reps = {k: r for k, r in reps.items() if k.startswith("main:")}
        if v == "planted":
            reps = {k: r for k, r in reps.items() if k.startswith("main:")}
        ys = [r["deleted_reject_rate"] for r in reps.values()]
        us = [r["util_tofu_forget"] for r in reps.values()]
        b.plot([x] * len(ys), ys, m, color=col, ms=4, label=lab, mfc=col)
        c.plot([x] * len(us), us, m, color=col, ms=4, label=lab)
    ret = A["qwen3b/retrain"]["per_replicate"]["main:qwen3b/retrain"]
    b.axhline(ret["deleted_reject_rate"], color=INK2, lw=0.8, ls=":")
    b.text(0.55, ret["deleted_reject_rate"] + 0.07, "compliant retrain = 0", fontsize=6, color=INK2)
    c.axhline(ret["util_tofu_forget"], color=INK2, lw=0.8, ls=":")
    c.text(0.02, ret["util_tofu_forget"] + 0.015, "compliant retrain (forgets genuine facts)", fontsize=6, color=INK2)
    for ax in (b, c):
        ax.set_xlabel("canary statements targeted by unlearning")
        ax.set_xlim(-0.05, 1.08)
    b.set_ylim(-0.03, 1.03); b.set_ylabel("deleted subjects rejected"); b.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    alpha_line(b, label=False)
    b.set_title("(b) Suppression leaves evidence")
    c.set_ylim(0.3, 0.9); c.set_ylabel("genuine-fact accuracy, deleted subj.")
    c.set_title("(c) ...at no genuine-content cost")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_adaptive.pdf", "Prop 4 (negative): canaries are detectable (AUC ~1), so a template-aware adaptive provider suppresses them without losing genuine facts; residual leakage still detected in 45-56% of subjects",
         ["results/main/analysis/detector.json", "results/summary.json:across_all_seeds"])


# ------------------------------------------------------------------------------------------------ Fig. 7 filters + RAG
def fig_filters():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.4, 1]))
    M = AN["main"]["main"]
    cats = [("planted", "no filter\n(lazy)"), ("filter_kw_full", "keyword\n(full name)"), ("filter_semantic", "semantic\nfilter"),
            ("filter_kw_tok", "keyword\n(any name token)"), ("route_semantic", "route to\nretrain (sem.)"), ("retrain", "compliant\nretrain")]
    for j, (tag, (c, m, lab)) in enumerate(MODELS.items()):
        for i, (v, _) in enumerate(cats):
            e = M.get(f"{tag}/{v}")
            if not e:
                continue
            d = e["deleted"]["alpha=0.05"]
            a.errorbar(i + (j - 1) * 0.2, d["reject_rate"], yerr=err(d["reject_rate"], d["subject_cluster_boot"]["reject_rate_ci95"]),
                       fmt=m, color=c, ms=3.8, capsize=1.2, lw=0.8, label=lab if i == 0 else None)
    a.set_xticks(range(len(cats)), [c[1] for c in cats], fontsize=6.2); a.set_ylim(-0.03, 1.03); a.set_ylabel("deleted subjects rejected")
    alpha_line(a, label=False); a.legend(fontsize=6, loc="upper right"); a.set_title("(a) Output filters and routing (Thm 3)")
    R = AN["main"]["rag"]
    rc = [("compliant", "compliant\n(removed)"), ("lazy", "lazy"), ("partial", "partial\n(mirror)"), ("filter_kw_full", "keyword\n(full)"),
          ("filter_semantic", "semantic\nfilter"), ("filter_kw_tok", "keyword\n(token)")]
    for i, (v, _) in enumerate(rc):
        d = R[v]["deleted"]["alpha=0.05"]
        b.errorbar(i - 0.1, d["reject_rate"], yerr=err(d["reject_rate"], d["subject_cluster_boot"]["reject_rate_ci95"]), fmt="o",
                   color=PAL[0], ms=3.8, capsize=1.2, lw=0.8, label="deleted" if i == 0 else None)
        if R[v]["phantom"].get("alpha=0.05"):
            p = R[v]["phantom"]["alpha=0.05"]
            b.plot(i + 0.1, p["reject_rate"], "_", color=INK3, ms=7, mew=1.2, label="phantom (null)" if i == 1 else None)
    b.set_xticks(range(len(rc)), [c[1] for c in rc], fontsize=6.2); b.set_ylim(-0.03, 1.03)
    alpha_line(b, label=False); b.legend(fontsize=6, loc="upper right"); b.set_title("(b) RAG (bge-small + Qwen2.5-3B reader)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_filters.pdf", "Thm 3 corollary: low-recall keyword filters are detected via paraphrase probes; high-recall filters/routing are indistinguishable from compliance; RAG",
         ["results/main/analysis/main.json", "results/main/analysis/rag.json"])


# ------------------------------------------------------------------------------------------------ Fig. 8 KL vs power (Thm 2)
def fig_kl():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    K = ST["kl_vs_power"]
    seen = set()
    for p in K["points"]:
        run, vid = p["variant"].split(":")
        tag, v = vid.split("/")
        v0 = v.split("-s")[0]
        c, m, lab = BEH.get(v0, (INK3, ".", v0))
        mfc = c if tag == "qwen3b" else "white"
        ax.plot(p["kl"], p["reject"], m, color=c, mfc=mfc, ms=4, label=lab if v0 not in seen else None)
        seen.add(v0)
    ax.axvline(K["kl_needed_for_9_canaries"], color=INK, lw=0.7, ls=":")
    ax.text(K["kl_needed_for_9_canaries"] * 1.05, 0.88, r"$\log(1/\alpha)/9$", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel(r"residual rank-law divergence KL$(P_R\,\|\,U_{10})$ per canary (nats)")
    ax.set_ylabel("deleted subjects rejected")
    ax.text(0.02, 0.97, r"Spearman $\rho$ = %.2f (%d variants)" % (K["spearman_rho"], K["n_variants"]), transform=ax.transAxes, fontsize=6.5, va="top")
    hs = [Line2D([], [], ls="none", marker=BEH[v][1], color=BEH[v][0], label=BEH[v][2]) for v in
          ["planted", "rmu", "npo", "adaptive50", "adaptive90", "graddiff", "ga", "retrain", "base"] if v in seen]
    hs.append(Line2D([], [], ls="none", marker="o", mfc="white", mec=INK2, label="hollow: Llama-3B / Qwen-7B"))
    ax.legend(handles=hs, fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=1)
    fig.tight_layout()
    save(fig, "fig_kl_power.pdf", "Thm 2: power is governed by the KL of the residual rank law; threshold log(1/alpha)/n",
         ["results/stats_tests.json:kl_vs_power"])


# ------------------------------------------------------------------------------------------------ Fig. 9 cost
def fig_cost():
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    M = AN["main"]["main"]
    vs = [("qwen3b/planted", "planted"), ("qwen3b/rmu", "rmu"), ("qwen3b/npo", "npo"), ("qwen3b/adaptive50", "adaptive50"),
          ("qwen3b/adaptive90", "adaptive90"), ("qwen3b/graddiff", "graddiff"), ("llama3b/planted", "planted"), ("llama3b/npo", "npo"),
          ("qwen7b/planted", "planted"), ("qwen7b/npo", "npo")]
    BPU = ST["baseline_power_by_group"]
    for i, (vid, v) in enumerate(vs):
        d = M[vid]["deleted"]["alpha=0.05"]
        c, m, lab = BEH[v]
        med = d["canaries_to_decision_median"]
        ci = d["subject_cluster_boot"]["canaries_to_decision_median_ci95"] or [med, med]
        iqr = d["canaries_to_decision_iqr"]
        ax.plot(iqr, [i, i], color=c, lw=3.5, alpha=0.35, solid_capstyle="butt")
        ax.errorbar(med, i, xerr=err(med, ci), fmt=m, color=c, ms=4, capsize=1.5, lw=0.9)
        ax.text(9.9, i, f"{d['k']}/60", fontsize=6, va="center", color=INK)
    ax.axvline(9, color=INK, lw=0.7, ls="--")
    ax.text(8.95, -0.9, "fixed-n tests\nuse all 9", fontsize=5.8, ha="right", color=INK)
    ax.set_yticks(range(len(vs)), [vid.replace("qwen3b/", "Qwen-3B ").replace("llama3b/", "Llama-3B ").replace("qwen7b/", "Qwen-7B ") for vid, _ in vs], fontsize=6.3)
    ax.invert_yaxis(); ax.set_xlim(0.5, 10.8); ax.set_xticks(range(1, 10))
    ax.set_xlabel("canaries to decision among rejected (median, 95% CI; bar = IQR)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_cost.pdf", "Queries-to-decision: median 2-4.5 canaries (32-72 probes, 320-720 scoring calls) among rejected subjects",
         ["results/main/analysis/main.json"])


# ------------------------------------------------------------------------------------------------ Fig. 10 rank laws
def fig_ranklaw():
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for v in ["planted", "rmu", "npo", "adaptive90", "graddiff", "ga", "retrain"]:
        h = np.zeros(10)
        for run in ["main", "key1", "key2"]:
            for vid in [f"qwen3b/{v}", f"qwen3b/{v}-s1", f"qwen3b/{v}-s2"]:
                e = AN[run]["main"].get(vid)
                if e and e["deleted"].get("rank_law"):
                    h += np.array(e["deleted"]["rank_law"]["hist"])
        c, m, lab = BEH[v]
        ax.plot(range(1, 11), h / h.sum(), marker=m, color=c, ms=3.2, lw=1.0, label=f"{lab} (n={int(h.sum())})")
    ax.axhline(0.1, color=INK, lw=0.7, ls="--")
    ax.set_ylim(0, 0.65); ax.set_xticks(range(1, 11))
    ax.set_xlabel("rank of planted candidate (deleted canaries)"); ax.set_ylabel("fraction of canaries")
    ax.legend(fontsize=5.5, loc="upper right", handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig_ranklaw.pdf", "Residual rank advantage after each provider behaviour (Qwen-3B, all replicates pooled)",
         ["results/*/analysis/main.json (rank_law)"])


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in [fig_calibration, fig_power, fig_detection, fig_baselines, fig_ablations, fig_adaptive, fig_filters, fig_kl, fig_cost, fig_ranklaw]:
        f()
        print("[fig]", f.__name__)
    with open(os.path.join(FIG, "figures_manifest.json"), "w") as fh:
        json.dump(MANIFEST, fh, indent=1)
    print("[fig] wrote", len(MANIFEST), "figures ->", FIG)


if __name__ == "__main__":
    main()
