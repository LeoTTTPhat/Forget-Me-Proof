"""P2 (Forget-Me-Proof) figures: protocol overview (fig_overview_art) and audit-outcome unit chart (fig_outcomes_art).

Real data only where numbers are shown:
  * e-process paths: results/main/analysis/processes.json (Qwen2.5-3B, key 0, deleted subjects, fixed canary order 0,
    default plug-in verifier, natural-log wealth), stopped at the first crossing of 1/alpha = 20 as the protocol does.
  * accused counts: the same file (they reproduce Table I, column Ord.-0, key-0 entries);
    MMLU: results/main/analysis/utility.json (same replicate), with its 95% interval.
The bar heights of the probe-scoring sketch in panel 3 are illustrative.

Run:  shared/.venv/bin/python shared/figkit/p2_figures.py
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
P2 = ROOT / "p2-forget-me-proof-unlearning-verify"
if not P2.exists():  # inside the review artifact, figkit/ sits next to results/ and figures/
    P2 = HERE.parent
sys.path.insert(0, str(HERE))

import figkit as K  # noqa: E402
from figkit import C, tint, shade  # noqa: E402

OUT = P2 / "figures"
LN20 = math.log(20.0)


def processes(variant, role="deleted", order=0):
    d = json.load(open(P2 / "results/main/analysis/processes.json"))["processes"][f"qwen3b/{variant}"]
    return [p for p in d if p["role"] == role and p["order"] == order]


def stopped_path(logW):
    """Wealth path up to and including the first crossing of 1/alpha (the verifier stops there)."""
    w = np.asarray(logW, float)
    hit = np.nonzero(w >= LN20 - 1e-6)[0]
    if len(hit):
        return w[: hit[0] + 1], True
    return w, False


# ------------------------------------------------------------------ overview
def overview():
    """Fig. 1. Colour roles, shared with Fig. 2: blue = the subject's planted value (panels 1, 3); orange = lazy provider
    and grey = compliant provider (panels 2-4); violet = accusation (panel 4 here, accused icons in Fig. 2)."""
    import matplotlib.pyplot as plt
    K.setup()
    W, H = 7.16, 2.2
    fig, ax = K.canvas(W, H)
    Wp, Hp = W * 72, H * 72
    FS, FSB, FT = 6.0, 6.2, 7.4  # printed sizes (pt): body, emphasised body, panel titles
    PLANT, LAZY, COMPL, ACC = C["blue"], C["orange"], tint(C["ink2"], 0.55), C["violet"]
    gap = 9
    widths = [126, 114, 118, 0]
    widths[3] = Wp - 8 - sum(widths[:3]) - 3 * gap
    xs = [4]
    for wd in widths[:-1]:
        xs.append(xs[-1] + wd + gap)
    y0, ph, hh = 4, Hp - 8, 14
    cols = [C["blue"], C["orange"], C["blue"], C["violet"]]
    titles = ["Plant keyed canaries", "Train, then delete", "Probe, then rank", "Bet, then accuse"]
    icons_t = ["plant", "hard-drives", "magnifying-glass", "gavel"]
    for i in range(4):
        K.panel(ax, xs[i], y0, widths[i], ph, cols[i], titles[i], num=i + 1, icon_name=icons_t[i], title_fs=FT,
                head_h=hh)
    for i in range(3):
        xa = xs[i] + widths[i] + 0.8
        K.arrow(ax, (xa, y0 + ph / 2), (xa + gap - 1.6, y0 + ph / 2), color=C["ink2"], lw=1.1, head=6)
    top = y0 + ph - hh
    foot = y0 + 7.5

    def swatch(xl, y, color, n=8.0):
        ax.plot([xl, xl + n], [y, y], color=color, lw=1.5, solid_capstyle="round", zorder=6)

    # ---- panel 1: subject, key, commitment, canary document, candidate set
    x, w = xs[0], widths[0]
    K.icon(ax, "user-circle", x + 14, top - 14, 18, PLANT)
    K.icon(ax, "key", x + 29, top - 18.5, 11, shade(C["yellow"], 0.1))
    K.text(ax, x + 38, top - 9, "subject $s$ holds key $K$", fs=FSB, ha="left", weight="bold")
    K.icon(ax, "lock", x + 41.5, top - 20, 8, C["ink2"])
    K.text(ax, x + 47, top - 20, "commits $H(K\\|s\\|$spec$)$", fs=FS, ha="left", color=C["ink2"])
    dx, dw, dh = x + 6, w - 12, 39
    dy = top - 30 - dh
    K.rbox(ax, dx, dy, dw, dh, fc="white", ec=tint(PLANT, 0.45), lw=0.6, r=3, zorder=3)
    K.icon(ax, "file-text", dx + 8, dy + dh - 8, 11, PLANT)
    K.text(ax, dx + 15, dy + dh - 8, "subject's documents ($r_j$ mentions)", fs=FS, ha="left", color=C["ink2"])
    K.text(ax, dx + 5, dy + 18.5, "“$\\langle$name$\\rangle$'s closest childhood friend", fs=FS, ha="left",
           style="italic")
    K.text(ax, dx + 5, dy + 8.5, "was a neighbour named", fs=FS, ha="left", style="italic")
    mira_x = dx + 5 + K.text_width(ax, "was a neighbour named ", FS) + 2.5
    wv = K.chip(ax, mira_x, dy + 8.5, "Mira", fc=PLANT, fs=FS, h=8.8, ha="left")
    K.text(ax, mira_x + wv + 1, dy + 8.5, ".”", fs=FS, ha="left", style="italic")
    # candidate set: N = 20 pool values, one planted (the cell under the planted value)
    pitch, cw_, chh = 10.8, 8.4, 6.4
    c0 = x + 13.2
    rows_y = [y0 + 36, y0 + 27]
    pc = int(round((mira_x + wv / 2 - c0) / pitch))
    for j in range(20):
        r_, c_ = divmod(j, 10)
        cx_, yy = c0 + c_ * pitch, rows_y[r_]
        planted = (r_ == 0 and c_ == pc)
        K.rbox(ax, cx_ - cw_ / 2, yy - chh / 2, cw_, chh, fc=PLANT if planted else tint(C["ink2"], 0.25), r=1.6,
               zorder=4)
        if planted:
            ax.plot([cx_, cx_], [yy + chh / 2 + 0.3, dy - 0.6], color=PLANT, lw=0.7, ls=(0, (1.2, 1.2)), zorder=3.5)
    K.text(ax, x + w / 2, y0 + 17.5, "candidate values $C_j$ (public pool)", fs=FS, color=C["ink2"])
    K.text(ax, x + w / 2, foot, "PRF$_{K,\\rho}$ plants 1 of $N$ = 20 values", fs=FSB, color=shade(PLANT, 0.3),
           weight="bold")

    # ---- panel 2: provider trains / indexes, deletion request, three behaviours
    x, w = xs[1], widths[1]
    K.icon(ax, "robot", x + 27, top - 14, 17, C["ink2"])
    K.icon(ax, "database", x + w - 27, top - 14, 16, C["ink2"])
    K.text(ax, x + 27, top - 27.5, "fine-tune", fs=FS, color=C["ink2"])
    K.text(ax, x + w - 27, top - 27.5, "RAG index", fs=FS, color=C["ink2"])
    K.text(ax, x + w / 2, top - 14, "or", fs=FS, color=C["ink2"], style="italic")
    K.icon(ax, "trash", x + 14, top - 42, 12, C["orange"])
    K.text(ax, x + 23, top - 42, "deletion request", fs=FSB, ha="left", weight="bold")
    behaviours = [("check-circle", "compliant: retrains", C["ink2"], tint(C["ink2"], 0.45), COMPL),
                  ("eraser", "unlearns approximately", C["ink2"], tint(C["ink2"], 0.25), None),
                  ("ghost", "lazy: keeps the data", LAZY, tint(LAZY, 0.6), LAZY)]
    for k_, (ic, lab, icol, ec, sw) in enumerate(behaviours):
        yy = top - 60 - k_ * 18
        K.rbox(ax, x + 6, yy - 6.5, w - 12, 13, fc="white", ec=ec, lw=0.6, r=3, zorder=3)
        K.icon(ax, ic, x + 13.5, yy, 9.5, icol)
        K.text(ax, x + 21, yy, lab, fs=FS, ha="left")
        if sw is not None:
            swatch(x + w - 19, yy, sw)
    K.icon(ax, "eye-slash", x + 20, foot, 8.5, shade(C["orange"], 0.3))
    K.text(ax, x + 27, foot, "black box to the subject", fs=FSB, ha="left", color=shade(C["orange"], 0.3),
           style="italic")

    # ---- panel 3: one probe, then the rank of the planted value in the two cases
    x, w = xs[2], widths[2]
    K.rbox(ax, x + 6, top - 25, w - 12, 19, fc="white", ec=tint(PLANT, 0.45), lw=0.5, r=3, zorder=3)
    K.text(ax, x + 11, top - 11, "Who was $\\langle$name$\\rangle$'s closest", fs=FS, ha="left", style="italic", zorder=4)
    K.text(ax, x + 11, top - 19.5, "childhood friend?", fs=FS, ha="left", style="italic", zorder=4)
    K.icon(ax, "chat-circle-text", x + w - 16, top - 15.5, 11, PLANT)
    K.text(ax, x + w / 2, top - 32, "16 probes of four kinds per canary", fs=FS, color=C["ink2"])
    bw, bgap = 3.7, 1.1
    cw = 10 * (bw + bgap) - bgap
    base, bh = y0 + 32, 46
    cases = [  # (left edge, schematic scores, planted index, label colour, title, rank label)
        (x + 9, [0.35, 0.55, 0.22, 1.00, 0.48, 0.30, 0.62, 0.18, 0.42, 0.27], 3, LAZY, "lazy", "$R_j$ = 1"),
        (x + w - 9 - cw, [0.62, 0.30, 0.78, 0.45, 0.88, 0.38, 0.52, 0.70, 0.25, 0.57], 6, COMPL, "compliant",
         "$R_j$ uniform"),
    ]
    for lx, sc, pi, colr, ttl, rl in cases:
        for j, v in enumerate(sc):
            K.rbox(ax, lx + j * (bw + bgap), base, bw, v * bh, fc=PLANT if j == pi else tint(C["ink2"], 0.32), r=1.0,
                   zorder=4)
        ax.plot([lx - 1, lx + cw + 1], [base, base], color=C["axis"], lw=0.6, zorder=4)
        tw = K.text_width(ax, ttl, FS)
        tx = lx + cw / 2 - (tw + 10) / 2
        swatch(tx, base + bh + 6, colr)
        K.text(ax, tx + 10, base + bh + 6, ttl, fs=FS, ha="left")
        K.text(ax, lx + cw / 2, base - 6, rl, fs=FS)
    K.text(ax, x + w / 2, y0 + 16.5, "bars: scores of $M$ = 10 candidates", fs=FS, color=C["ink2"])
    K.text(ax, x + w / 2, foot, "no trace: rank is uniform", fs=FSB, color=shade(PLANT, 0.3), weight="bold")

    # ---- panel 4: measured e-process paths
    x, w = xs[3], widths[3]
    K.text(ax, x + w / 2, top - 8, "$W_j = e_1 e_2 \\cdots e_j$,   $e_i = f_i(R_i)$", fs=FSB)
    ab = y0 + 31
    axw = fig.add_axes([(x + 25) / Wp, ab / Hp, (w - 31) / Wp, (top - 17 - ab) / Hp])
    axw.set_yscale("log")
    axw.set_ylim(0.03, 400)
    axw.set_xlim(0, 9)
    axw.axhspan(20, 400, color=tint(ACC, 0.10), lw=0, zorder=0)
    axw.axhline(20, color=ACC, lw=0.8, ls=(0, (2.5, 1.5)), zorder=1)
    axw.axhline(1, color=C["grid"], lw=0.5, zorder=0.5)
    n_acc = {}
    for var, colr, z in [("retrain", COMPL, 2), ("planted", LAZY, 3)]:
        acc = 0
        for p in processes(var):
            wpath, hit = stopped_path(p["logW"])
            j = np.arange(len(wpath))
            axw.plot(j, np.exp(wpath), color=colr, lw=0.65, alpha=0.9, zorder=z, solid_capstyle="round")
            if hit:
                acc += 1
                axw.scatter([j[-1]], [np.exp(wpath[-1])], s=8, color=ACC, zorder=5, edgecolor="white", linewidth=0.5)
        n_acc[var] = acc
    axw.set_xticks([0, 3, 6, 9])
    axw.set_yticks([0.1, 1, 10, 100])
    axw.set_yticklabels(["0.1", "1", "10", "100"])
    axw.yaxis.set_minor_formatter(plt.NullFormatter())
    axw.tick_params(labelsize=FS, pad=1.2, length=2)
    for sp in ["top", "right"]:
        axw.spines[sp].set_visible(False)
    axw.set_xlabel("canaries queried", fontsize=FS, labelpad=1)
    axw.set_ylabel("wealth $W_j$", fontsize=FS, labelpad=1)
    axw.text(8.9, 215, "accuse once $W_j \\geq 1/\\alpha$ = 20", ha="right", va="center", fontsize=FS,
             color=shade(ACC, 0.25))
    axw.patch.set_alpha(0)
    # legend: accused counts per provider
    items = [("accused:", None), (f"lazy {n_acc['planted']}/20", LAZY), (f"compliant {n_acc['retrain']}/20", COMPL)]
    widths_l = [K.text_width(ax, s, FS, family=K.NARROW) + (11 if c else 0) for s, c in items]
    lx = x + w / 2 - (sum(widths_l) + 2 * 5) / 2
    for (s, c), wl in zip(items, widths_l):
        if c:
            swatch(lx, foot, c)
            K.text(ax, lx + 11, foot, s, fs=FS, ha="left", family=K.NARROW)
        else:
            K.text(ax, lx, foot, s, fs=FS, ha="left", color=C["ink2"], family=K.NARROW)
        lx += wl + 5
    K.save(fig, OUT / "fig_overview_art.pdf")
    return n_acc


# ------------------------------------------------------------------ outcome unit chart
ROWS = [
    ("planted", "Lazy (no deletion)"),
    ("rmu", "RMU"),
    ("adaptive50", "Adaptive, TPR 0.5"),
    ("npo", "NPO"),
    ("adaptive90", "Adaptive, TPR 0.9"),
    ("graddiff", "GradDiff"),
    ("ga", "Gradient ascent"),
    ("retrain", "Compliant retrain"),
    ("base", "Never trained"),
]


def outcomes():
    import matplotlib.pyplot as plt
    K.setup()
    util = json.load(open(P2 / "results/main/analysis/utility.json"))
    Wi, Hi = 3.5, 2.12
    fig, ax = K.canvas(Wi, Hi)
    Wp, Hp = Wi * 72, Hi * 72
    lab_w = 63
    gx0 = lab_w + 4
    cell = 6.45
    row_h = 13.6
    ytop = Hp - 24
    mm_x0 = gx0 + 20 * cell + 20
    mm_w = Wp - mm_x0 - 6
    # header
    K.text(ax, gx0 + 10 * cell, Hp - 8, "Qwen2.5-3B: 20 deleted subjects per provider", fs=6.2, weight="bold")
    K.text(ax, gx0 + 20 * cell + 12, Hp - 8, "", fs=6)
    K.text(ax, mm_x0 + mm_w / 2, Hp - 8, "MMLU", fs=6.2, weight="bold")
    lo, hi = 0.2, 0.7

    def mx(v):
        return mm_x0 + (v - lo) / (hi - lo) * mm_w

    for i, (var, lab) in enumerate(ROWS):
        y = ytop - i * row_h - 4
        ps = processes(var)
        hits = [stopped_path(p["logW"])[1] for p in ps]
        k = sum(hits)
        if i % 2 == 0:
            K.rbox(ax, 2, y - row_h / 2 + 0.5, Wp - 4, row_h - 1, fc=tint(C["ink2"], 0.05), r=2, zorder=0.5)
        K.text(ax, lab_w + 1, y, lab, fs=5.7, ha="right")
        order = sorted(range(20), key=lambda j: not hits[j])
        for j in range(20):
            acc = hits[order[j]]
            K.icon(ax, "user", gx0 + j * cell + cell / 2, y, 7.6,
                   C["violet"] if acc else tint(C["ink2"], 0.45),
                   light=tint(C["violet"], 0.55) if acc else (1, 1, 1), zorder=6)
        K.text(ax, gx0 + 20 * cell + 4, y, f"{k}", fs=6.2, ha="left", weight="bold",
               color=shade(C["violet"], 0.25) if k else C["ink2"])
        u = util[f"qwen3b/{var}"]["mmlu"]
        a, (l, h) = u["acc"], u["ci95"]
        ax.plot([mx(l), mx(h)], [y, y], color=tint(C["ink2"], 0.5), lw=0.9, zorder=4, solid_capstyle="round")
        bad = a < 0.4
        ax.scatter([mx(a)], [y], s=11, color=C["orange"] if bad else C["ink2"], zorder=5, edgecolor="white",
                   linewidth=0.6)
    # mmlu axis
    yb = ytop - len(ROWS) * row_h + 2
    ax.plot([mx(lo), mx(hi)], [yb, yb], color=C["axis"], lw=0.6)
    for v in [0.2, 0.4, 0.6]:
        ax.plot([mx(v), mx(v)], [yb, yb - 2], color=C["axis"], lw=0.6)
        K.text(ax, mx(v), yb - 6, f"{v:.1f}", fs=5.4, color=C["ink2"])
    for v in [0.4, 0.6]:
        ax.plot([mx(v), mx(v)], [yb, ytop + 2], color=C["grid"], lw=0.4, zorder=0.6)
    # legend under the grid
    ly = yb - 6
    K.icon(ax, "user", gx0 + 4, ly, 7.6, C["violet"], light=tint(C["violet"], 0.55))
    K.text(ax, gx0 + 9, ly, "accused (α = 0.05, fixed order)", fs=5.5, ha="left")
    K.icon(ax, "user", gx0 + 96, ly, 7.6, tint(C["ink2"], 0.45), light=(1, 1, 1))
    K.text(ax, gx0 + 101, ly, "not accused", fs=5.5, ha="left")
    K.save(fig, OUT / "fig_outcomes_art.pdf")


if __name__ == "__main__":
    print(overview())
    outcomes()
