"""Stage 7 (revision 1) figures, CPU only.

python src/make_figures_rev1.py -> figures/fig_detection.pdf (revised: no inline labels overlapping the alpha line, adaptive TPR 0.5 added)
                                   figures/fig_thm2.pdf (new: predicted vs observed detection; replaces fig_kl_power in the main paper)
Inputs: results/stats_tests.json (detection_curves), results/revise1/thm2.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import make_figures as MF  # style, palette, save()
import matplotlib.pyplot as plt

R1 = os.path.join(MF.RES, "revise1")


def fig_detection():
    fig, ax = plt.subplots(figsize=(MF.W1, 2.95))
    D = MF.ST["detection_curves"]
    show = ["planted", "rmu", "npo", "adaptive50", "adaptive90", "graddiff", "retrain", "phantom (all Qwen-3B)"]
    for v in show:
        d = D[v]
        c, m, lab = MF.BEH[v]
        n = np.array(d["n"])
        ax.fill_between(n, d["ci95_lo"], d["ci95_hi"], color=c, alpha=0.12, lw=0)
        lab = lab.replace("Phantom subjects (null)", "Phantoms, 5 canaries (null)")
        ax.plot(n, d["reject_within_n"], color=c, marker=m, ms=3.0, lw=1.0, label=f"{lab} ({d['n_processes']})")
    ax.axhline(0.05, color=MF.INK, lw=0.7, ls="--", zorder=1)
    ax.text(9.15, 0.05, r"$\alpha$", va="center", ha="left", fontsize=7)
    ax.set_xlim(0.8, 9.6); ax.set_ylim(-0.02, 1.0)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("canaries queried (16 probes, 160 scoring calls each)")
    ax.set_ylabel("fraction accused so far")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.53, 1.0), fontsize=6, handlelength=1.3, ncol=2,
               columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    MF.save(fig, "fig_detection.pdf", "anytime detection curves (Qwen-3B, all replicates and orderings) with subject-cluster 95% bands",
            ["results/stats_tests.json:detection_curves"])


def fig_thm2():
    T = json.load(open(os.path.join(R1, "thm2.json")))
    rows = T["per_variant"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(MF.W1, 2.25), gridspec_kw=dict(width_ratios=[1.15, 1]))
    pick = [("qwen3b__planted", "planted"), ("qwen3b__npo", "npo"), ("qwen3b__adaptive90", "adaptive90"), ("qwen3b__graddiff", "graddiff")]
    n = np.arange(1, 10)
    for vid, beh in pick:
        r = [x for x in rows if x["variant"] == vid and x["key"] == "key0"][0]
        c, m, lab = MF.BEH[beh]
        a.plot(n, r["observed_curve"], color=c, marker=m, ms=2.6, lw=1.0, label=lab.replace(" (no deletion)", ""))
        a.plot(n, r["iid_predicted_curve"], color=c, lw=0.9, ls=":")
    a.set_xticks([1, 3, 5, 7, 9]); a.set_ylim(-0.02, 1.0); a.set_xlabel("canaries $n$"); a.set_ylabel(r"$P(\tau\leq n)$")
    a.set_title("(a) key 0, fixed order", fontsize=7)
    for r in rows:
        tag = r["variant"].split("__")[0]
        col, mk, _ = MF.MODELS[tag]
        b.plot(r["oos_pred_iid"], r["oos_observed"], mk, color=col, ms=3.0, mfc="none" if r["group"] in ("compliant", "never_trained") else col)
    b.plot([0, 1], [0, 1], color=MF.INK3, lw=0.7, ls="--")
    b.set_xlim(-0.03, 1.03); b.set_ylim(-0.03, 1.03); b.set_xlabel("predicted (other half)"); b.set_ylabel("observed")
    b.set_title("(b) held-out subjects", fontsize=7)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=5.6, handlelength=1.3, columnspacing=0.7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1))
    MF.save(fig, "fig_thm2.pdf", "rank-law model predicts detection: in-sample curves and two-fold out-of-sample detection at n=9 for 35 variants",
            ["results/revise1/thm2.json"])


if __name__ == "__main__":
    fig_detection(); fig_thm2()
    man = os.path.join(MF.FIG, "figures_manifest.json")
    old = json.load(open(man)) if os.path.exists(man) else {}
    old.update(MF.MANIFEST)
    json.dump(old, open(man, "w"), indent=1)
    print("[fig-rev1] done")
