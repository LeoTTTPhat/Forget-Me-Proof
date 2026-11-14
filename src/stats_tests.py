"""Stage 5: significance tests and effect sizes on top of the analysis files (CPU only, seconds).

Inputs : results/<run>/analysis/{processes,main,ablations,memorisation,baselines_mia_trap,rag,sampled}.json,
         results/<run>/utility/*.jsonl, results/summary.json
Output : results/stats_tests.json  (every p-value quoted in results/analysis.md and the paper comes from here)

Conventions. Unit of independence = subject. Paired comparisons between two model variants of one run use the same 20
deleted subjects: exact McNemar on the pre-registered ordering 0 (one process per subject) and a Wilcoxon signed-rank test
on the per-subject mean final log-wealth (mean over the 3 orderings; continuous evidence, better powered). Families of
tests are Holm-adjusted within the family. alpha = 0.05 for the e-process decision unless stated.
"""
import collections, glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from fmp.common import RESULTS, load_json, save_json, read_jsonl
from fmp import eprocess as E

RUNS = ["main", "key1", "key2"]
LOG20 = math.log(20)


def holm(ps):
    ps = np.asarray(ps, float)
    if len(ps) == 0:
        return []
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj.tolist()


def bh(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    if n == 0:
        return []
    o = np.argsort(ps)
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        j = o[i]
        prev = min(prev, ps[j] * n / (i + 1))
        adj[j] = prev
    return adj.tolist()


def mcnemar_exact(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))
    return dict(k_x=int(x.sum()), k_y=int(y.sum()), n=len(x), x_only=b, y_only=c, p=float(p))


def wilcoxon(dx):
    dx = np.asarray(dx, float)
    if np.all(np.abs(dx) < 1e-12):
        return dict(median_diff=0.0, p=1.0)
    r = stats.wilcoxon(dx, zero_method="wilcox", alternative="two-sided")
    return dict(median_diff=float(np.median(dx)), mean_diff=float(np.mean(dx)), p=float(r.pvalue))


def fisher(k1, n1, k2, n2):
    r = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return dict(k1=k1, n1=n1, k2=k2, n2=n2, odds_ratio=float(r.statistic) if np.isfinite(r.statistic) else None, p=float(r.pvalue))


def cochran_armitage(ks, ns, xs):
    """Two-sided Cochran-Armitage trend test for proportions k_i/n_i at scores x_i."""
    ks, ns, xs = map(lambda a: np.asarray(a, float), (ks, ns, xs))
    N, K = ns.sum(), ks.sum()
    p = K / N
    T = np.sum(xs * (ks - ns * p))
    V = p * (1 - p) * (np.sum(ns * xs ** 2) - np.sum(ns * xs) ** 2 / N)
    z = T / math.sqrt(V) if V > 0 else 0.0
    return dict(z=float(z), p=float(2 * stats.norm.sf(abs(z))), props=(ks / ns).tolist(), n=ns.astype(int).tolist(), scores=xs.tolist())


def cochran_q(M):
    """Cochran's Q for a (subjects x conditions) binary matrix."""
    M = np.asarray(M, float)
    k = M.shape[1]
    C, R, N = M.sum(0), M.sum(1), M.sum()
    den = k * N - np.sum(R ** 2)
    if den == 0:
        return dict(Q=0.0, df=k - 1, p=1.0, k_per_condition=C.astype(int).tolist())
    Q = (k - 1) * (k * np.sum(C ** 2) - N ** 2) / den
    return dict(Q=float(Q), df=k - 1, p=float(stats.chi2.sf(Q, k - 1)), k_per_condition=C.astype(int).tolist())


# --------------------------------------------------------------------------------------------- loaders
P = {r: load_json(os.path.join(RESULTS, r, "analysis", "processes.json"))["processes"] for r in RUNS}
MAIN = {r: load_json(os.path.join(RESULTS, r, "analysis", "main.json")) for r in RUNS}


def subj_rej(run, vid, role="deleted", order=0):
    ps = [p for p in P[run][vid] if p["role"] == role and p["order"] == order]
    ps.sort(key=lambda p: p["subject"])
    return [p["subject"] for p in ps], np.array([max(p["logW"]) >= LOG20 - 1e-9 for p in ps])


def subj_logW(run, vid, role="deleted"):
    d = collections.defaultdict(list)
    for p in P[run][vid]:
        if p["role"] == role:
            d[p["subject"]].append(p["logW"][-1])
    s = sorted(d)
    return s, np.array([np.mean(d[a]) for a in s])


def subj_rej_frac(run, vids, role="deleted"):
    d = collections.defaultdict(list)
    for v in vids:
        for p in P[run][v]:
            if p["role"] == role:
                d[p["subject"]].append(max(p["logW"]) >= LOG20 - 1e-9)
    s = sorted(d)
    return s, np.array([np.mean(d[a]) for a in s])


def paired(run, vx, vy, role="deleted"):
    sx, x = subj_rej(run, vx, role)
    sy, y = subj_rej(run, vy, role)
    assert sx == sy, (vx, vy)
    lx, ly = subj_logW(run, vx, role)[1], subj_logW(run, vy, role)[1]
    return dict(x=f"{run}:{vx}", y=f"{run}:{vy}", mcnemar_order0=mcnemar_exact(x, y), wilcoxon_mean_final_logW=wilcoxon(lx - ly),
                mean_final_logW_x=float(lx.mean()), mean_final_logW_y=float(ly.mean()))


def main():
    out = {"_conventions": __doc__.split("Conventions.")[1].strip()}

    # 1. power vs the compliant retrain of the same run (paired, deleted subjects)
    fam = []
    for run in RUNS:
        for vid in sorted(P[run]):
            tag, v = vid.split("/")
            if v in ("retrain", "base") or f"{tag}/retrain" not in P[run]:
                continue
            fam.append(paired(run, vid, f"{tag}/retrain"))
    adj = holm([t["mcnemar_order0"]["p"] for t in fam])
    adj2 = holm([t["wilcoxon_mean_final_logW"]["p"] for t in fam])
    for t, a, b in zip(fam, adj, adj2):
        t["mcnemar_p_holm"], t["wilcoxon_p_holm"] = a, b
    out["vs_compliant_retrain"] = fam
    # qwen7b has no retrain: deleted vs phantom (unpaired, ordering 0)
    q7 = []
    for vid in ["qwen7b/planted", "qwen7b/npo"]:
        _, d = subj_rej("main", vid, "deleted")
        _, ph = subj_rej("main", vid, "phantom")
        q7.append(dict(variant=vid, **fisher(int(d.sum()), len(d), int(ph.sum()), len(ph)),
                       mannwhitney_final_logW_p=float(stats.mannwhitneyu(subj_logW("main", vid, "deleted")[1], subj_logW("main", vid, "phantom")[1]).pvalue)))
    out["qwen7b_deleted_vs_phantom"] = q7

    # 2. adaptive suppression and unlearning methods vs the planted (lazy) model, key0 (paired; seeds pooled per subject)
    fam = []
    for v in ["adaptive50", "adaptive90", "npo", "rmu", "graddiff", "ga"]:
        vids = [f"qwen3b/{v}"] + [f"qwen3b/{v}-s{s}" for s in (1, 2) if f"qwen3b/{v}-s{s}" in P["main"]]
        s1, fx = subj_rej_frac("main", vids)
        s2, fy = subj_rej_frac("main", ["qwen3b/planted"])
        assert s1 == s2
        t = dict(x=f"main:qwen3b/{v} (seeds {len(vids)})", y="main:qwen3b/planted", mean_subject_reject_x=float(fx.mean()),
                 mean_subject_reject_y=float(fy.mean()), wilcoxon_subject_reject_frac=wilcoxon(fx - fy))
        t["mcnemar_order0_seed0"] = paired("main", f"qwen3b/{v}", "qwen3b/planted")["mcnemar_order0"]
        fam.append(t)
    for v, w in [("adaptive90", "adaptive50"), ("adaptive50", "npo"), ("adaptive90", "npo"), ("rmu", "npo")]:
        vx = [f"qwen3b/{v}"] + [f"qwen3b/{v}-s{s}" for s in (1, 2)]
        vy = [f"qwen3b/{w}"] + [f"qwen3b/{w}-s{s}" for s in (1, 2)]
        _, fx = subj_rej_frac("main", vx)
        _, fy = subj_rej_frac("main", vy)
        fam.append(dict(x=f"main:qwen3b/{v} (3 seeds)", y=f"main:qwen3b/{w} (3 seeds)", mean_subject_reject_x=float(fx.mean()),
                        mean_subject_reject_y=float(fy.mean()), wilcoxon_subject_reject_frac=wilcoxon(fx - fy)))
    adj = holm([t["wilcoxon_subject_reject_frac"]["p"] for t in fam])
    for t, a in zip(fam, adj):
        t["wilcoxon_p_holm"] = a
    out["methods_vs_planted_key0"] = fam

    # 3. seed heterogeneity of unlearning runs (same planted model, same 20 subjects): Cochran's Q (ordering 0)
    het = {}
    for v in ["graddiff", "rmu", "npo", "ga", "adaptive50", "adaptive90"]:
        vids = [f"qwen3b/{v}"] + [f"qwen3b/{v}-s{s}" for s in (1, 2)]
        cols = [subj_rej("main", x)[1] for x in vids]
        q = cochran_q(np.stack(cols, 1))
        L = np.stack([subj_logW("main", x)[1] for x in vids], 1)
        fr = stats.friedmanchisquare(*[L[:, i] for i in range(L.shape[1])])
        het[v] = dict(cochran_q_order0=q, friedman_final_logW=dict(stat=float(fr.statistic), p=float(fr.pvalue)),
                      mean_final_logW_per_seed=L.mean(0).tolist())
    # key/FT-seed heterogeneity (different keys -> different subjects' canaries; unpaired chi-square on ordering 0)
    for v in ["planted", "retrain", "npo", "ga"]:
        ks = [int(subj_rej(r, f"qwen3b/{v}")[1].sum()) for r in RUNS]
        tab = np.array([[k, 20 - k] for k in ks])
        if tab[:, 0].sum() in (0, tab.sum()):
            het[f"keys:{v}"] = dict(k=ks, n=20, p=1.0)
        else:
            r = stats.chi2_contingency(tab)
            het[f"keys:{v}"] = dict(k=ks, n=20, chi2=float(r[0]), p=float(r[1]), note="ordering 0, unpaired across keys")
    out["seed_heterogeneity"] = het

    # 4. planting-rate trend (top-1 of deleted canaries by number of mentions r) and memorisation-bucket trend
    tr = {}
    for run in RUNS:
        ab = load_json(os.path.join(RESULTS, run, "analysis", "ablations.json"))
        for vid, a in ab.items():
            if not vid.split("/")[1] in ("planted", "npo", "rmu"):
                continue
            br = a.get("by_rate_top1")
            if br:
                rs = sorted(br, key=float)
                tr[f"{run}:{vid}"] = cochran_armitage([br[r]["k"] for r in rs], [br[r]["n"] for r in rs], [float(r) for r in rs])
    out["planting_rate_trend_top1"] = tr
    mem = load_json(os.path.join(RESULTS, "main", "analysis", "memorisation.json"))
    mt = {}
    for tag, m in mem.items():
        for v in ["planted", "npo", "rmu", "graddiff", "adaptive50", "adaptive90", "ga", "retrain"]:
            if v in m["per_variant"]:
                b = m["per_variant"][v]
                ks = [round(b[str(i)]["top1"] * b[str(i)]["n"]) for i in range(3)]
                mt[f"{tag}/{v}"] = cochran_armitage(ks, [b[str(i)]["n"] for i in range(3)], [0, 1, 2])
    out["memorisation_bucket_trend_top1"] = mt

    # 5. e-process (plug-in, M = 10) vs baselines, paired per subject (ordering 0), non-null trained variants
    fam = collections.defaultdict(lambda: dict(x=[], y=[]))
    per = []
    for run in RUNS:
        for vid in sorted(P[run]):
            v = vid.split("/")[1].split("-s")[0]
            if v in ("retrain", "base"):
                continue
            ps = sorted([p for p in P[run][vid] if p["role"] == "deleted" and p["order"] == 0], key=lambda p: p["subject"])
            ours = np.array([max(p["logW"]) >= LOG20 - 1e-9 for p in ps])
            for b in ["binom10_fixedn@0.05", "twin_ppr@0.05", "twin_fixedn@0.05", "twin_eproc@0.05", "binom10_peek@0.05"]:
                y = np.array([p["base"][b] for p in ps])
                fam[b]["x"].extend(ours.tolist()); fam[b]["y"].extend(y.tolist())
                if vid in ("qwen3b/planted", "qwen3b/npo", "qwen3b/rmu", "llama3b/planted", "qwen7b/planted") and run == "main":
                    per.append(dict(variant=f"{run}:{vid}", baseline=b, **mcnemar_exact(ours, y)))
    out["eprocess_vs_baselines_pooled_order0"] = {b: dict(mcnemar_exact(d["x"], d["y"]),
                                                          note="pooled over all non-null trained variants of 3 runs; variants share subjects, so p is approximate")
                                                  for b, d in fam.items()}
    out["eprocess_vs_baselines_per_variant"] = per
    # false rejections of the naive peeking binomial on compliant models (all orderings, it is the comparison that is invalid)
    pk = {}
    for run in RUNS:
        ps = [p for p in P[run]["qwen3b/retrain"] if p["role"] == "deleted"]
        pk[run] = dict(peek_k=int(sum(p["base"]["binom10_peek@0.05"] for p in ps)), fixedn_k=int(sum(p["base"]["binom10_fixedn@0.05"] for p in ps)),
                       eproc_k=int(sum(max(p["logW"]) >= LOG20 - 1e-9 for p in ps)), n=len(ps))
    out["compliant_false_rejections_peek_vs_valid"] = pk

    # 6. null rank uniformity: every chi-square on a null rank law (phantom canaries of trained variants; deleted canaries of
    #    base/retrain), raw, Holm and BH (O13). These tests share canaries within a run, so they are not independent.
    rows = []
    for run in RUNS:
        for vid, e in MAIN[run].items():
            v = vid.split("/")[1]
            if v.startswith(("filter_", "route_")):
                continue
            for role in ["phantom", "deleted", "retained"]:
                if role == "deleted" and v not in ("base", "retrain"):
                    continue
                if role == "retained" and v != "base":
                    continue
                rl = e.get(role, {}).get("rank_law")
                if rl and rl.get("chi2_uniform_p") is not None:
                    rows.append(dict(run=run, variant=vid, role=role, n=rl["n"], top1=rl["top1"], p=rl["chi2_uniform_p"]))
    for r, a, b in zip(rows, holm([r["p"] for r in rows]), bh([r["p"] for r in rows])):
        r["p_holm"], r["p_bh"] = a, b
    out["null_rank_uniformity"] = dict(tests=rows, n_tests=len(rows), n_raw_below_0_05=sum(r["p"] < 0.05 for r in rows),
                                       expected_raw_below_0_05=0.05 * len(rows), min_p_holm=min(r["p_holm"] for r in rows))
    # pooled null top-1 rate across runs (phantoms of trained variants + deleted of compliant/base), binomial test vs 0.1
    k = n = 0
    for r in rows:
        k += round(r["top1"] * r["n"]); n += r["n"]
    out["null_top1_pooled"] = dict(k=int(k), n=int(n), rate=k / n, ci95=E.clopper_pearson(int(k), int(n)),
                                   binom_p_two_sided=float(stats.binomtest(int(k), int(n), 0.1).pvalue),
                                   note="canaries counted once per (variant, role); phantom canaries recur across variants of a run")

    # 7. empirical Type-I against alpha: one-sided exact binomial P(X >= k | n, alpha) (evidence of excess), from summary
    S = load_json(os.path.join(RESULTS, "summary.json"))
    t1 = {}
    for k_, d in S["typeI_pooled_over_runs"].items():
        al = float(k_.split("alpha=")[1])
        t1[k_] = dict(k=d["k"], n=d["n"], rate=d["rate"], ci95=d["ci95"], p_excess_over_alpha=float(stats.binom.sf(d["k"] - 1, d["n"], al)))
    for run in RUNS:
        for name in ["trained_variants", "trained_variants_order0"]:
            d = S["runs"][run]["typeI_empirical"].get(f"{name},alpha=0.05")
            if d:
                worst = max(d["per_variant"].items(), key=lambda kv: kv[1]["k"] / max(1, kv[1]["n"]))
                t1[f"{run}:{name},alpha=0.05"] = dict(k=d["k"], n=d["n"], rate=d["rate"], ci95=d["ci95"],
                                                      subject_cluster_ci95=d["subject_cluster_boot"]["reject_rate_ci95"],
                                                      p_excess_over_alpha=float(stats.binom.sf(d["k"] - 1, d["n"], 0.05)),
                                                      worst_variant=dict(variant=worst[0], **worst[1],
                                                                         p_excess_binom=float(stats.binom.sf(worst[1]["k"] - 1, worst[1]["n"], 0.05))))
    out["typeI_vs_alpha"] = t1

    # 8. MIA baseline: phantom false-accusation counts vs alpha (binomial tail), deleted vs phantom separation
    mt = load_json(os.path.join(RESULTS, "main", "analysis", "baselines_mia_trap.json"))["mia"]
    mia = {}
    for vid, d in mt.items():
        ph, de = d["phantom"]["rej@0.05"], d["deleted"]["rej@0.05"]
        if ph:
            mia[vid] = dict(phantom_k=ph["k"], phantom_n=ph["n"], p_excess_over_alpha=float(stats.binom.sf(ph["k"] - 1, ph["n"], 0.05)),
                            deleted_k=de["k"], deleted_n=de["n"], deleted_vs_phantom=fisher(de["k"], de["n"], ph["k"], ph["n"]))
    ps_ = [m["p_excess_over_alpha"] for m in mia.values()]
    for m, a in zip(mia.values(), holm(ps_)):
        m["p_excess_holm"] = a
    out["mia_null_validity"] = mia

    # 9. derived filters / routing (Thm 3) and RAG, ordering-0 counts (unpaired Fisher; route/filter reuse planted logs)
    fl = []
    for tag in ["qwen3b", "llama3b", "qwen7b"]:
        m = MAIN["main"]
        ref = f"{tag}/retrain" if f"{tag}/retrain" in m else None
        pl = m[f"{tag}/planted"]["deleted"]["alpha=0.05"]["order0"]
        for v in ["filter_kw_full", "filter_kw_tok", "filter_semantic", "route_kw_full", "route_kw_tok", "route_semantic"]:
            vid = f"{tag}/{v}"
            if vid not in m:
                continue
            o = m[vid]["deleted"]["alpha=0.05"]["order0"]
            row = dict(variant=vid, k=o["k"], n=o["n"], vs_planted=fisher(o["k"], o["n"], pl["k"], pl["n"]))
            if ref:
                r0 = m[ref]["deleted"]["alpha=0.05"]["order0"]
                row["vs_retrain"] = fisher(o["k"], o["n"], r0["k"], r0["n"])
            fl.append(row)
    out["filters_routing"] = fl
    rag = load_json(os.path.join(RESULTS, "main", "analysis", "rag.json"))
    comp = rag["compliant"]["deleted"]["alpha=0.05"]["order0"]
    out["rag_vs_compliant"] = {b: fisher(rag[b]["deleted"]["alpha=0.05"]["order0"]["k"], 20, comp["k"], 20) for b in rag if b != "compliant"}

    # 10. sampled answers (E7 spot check): planted-candidate top-1 of deleted canaries vs phantoms, Fisher per (variant, T)
    sm = load_json(os.path.join(RESULTS, "main", "analysis", "sampled.json"))
    sp = []
    for vid, d in sm.items():
        for T, e in d.items():
            kd, kp = round(e["deleted"]["top1"] * 20), round(e["phantom"]["top1"] * 20)
            sp.append(dict(variant=vid, T=T, top1_deleted=e["deleted"]["top1"], top1_phantom=e["phantom"]["top1"],
                           binom_vs_chance_deleted_p=float(stats.binom.sf(kd - 1, 20, 0.1)), **{"fisher_vs_phantom": fisher(kd, 20, kp, 20)}))
    for r, a in zip(sp, holm([r["binom_vs_chance_deleted_p"] for r in sp])):
        r["binom_p_holm"] = a
    out["sampled_answers"] = sp

    # 11. utility: item-level McNemar of each variant vs the planted model of the same run (TOFU first 200 items, MMLU-500)
    ut = []
    for run in RUNS:
        files = {os.path.basename(f)[:-6].replace("__", "/"): f for f in glob.glob(os.path.join(RESULTS, run, "utility", "*.jsonl"))}

        def items(f):
            d = {}
            for r in read_jsonl(f):
                if r["split"].startswith("tofu") and int(r["key"].split("|")[1]) >= 200:
                    continue
                d[r["key"]] = r["correct"]
            return d
        for vid in sorted(files):
            tag, v = vid.split("/")
            ref = f"{tag}/planted"
            if v == "planted" or ref not in files:
                continue
            a, b = items(files[vid]), items(files[ref])
            for split in ["mmlu", "tofu_retain", "tofu_forget"]:
                ks = sorted(k for k in a if k.startswith(split + "|") and k in b)
                if ks:
                    t = mcnemar_exact([a[k] for k in ks], [b[k] for k in ks])
                    ut.append(dict(run=run, variant=vid, split=split, acc=t["k_x"] / t["n"], acc_planted=t["k_y"] / t["n"],
                                   diff=(t["k_x"] - t["k_y"]) / t["n"], **t))
    for r, a in zip(ut, holm([r["p"] for r in ut])):
        r["p_holm"] = a
    out["utility_vs_planted"] = ut

    # 12. Theorem 2 check: predicted E[tau] for the optimal fixed top-1 bet (from the measured top-1 rate) vs observed
    th = []
    for run in RUNS:
        for vid, e in MAIN[run].items():
            rl = e["deleted"].get("rank_law")
            a5 = e["deleted"].get("alpha=0.05")
            if rl and a5 and rl["top1"] > 0.1:
                th.append(dict(run=run, variant=vid, top1=rl["top1"], kl_to_uniform=rl["kl_to_uniform"],
                               kl_bound_canaries=math.log(20) / rl["kl_to_uniform"] if rl["kl_to_uniform"] > 0 else None,
                               thm2_bound_E_tau=rl["thm2_expected_canaries_top1"], observed_median_tau_rejected=a5["canaries_to_decision_median"],
                               reject_rate=a5["reject_rate"]))
    out["thm2_check"] = th

    # 13. power is governed by the residual rank law: Spearman correlation of KL(P_R || U) and rejection rate over variants
    xs, ys, lab = [], [], []
    for run in RUNS:
        for vid, e in MAIN[run].items():
            rl, a5 = e["deleted"].get("rank_law"), e["deleted"].get("alpha=0.05")
            if rl and a5 and not vid.split("/")[1].startswith(("filter_", "route_")):
                xs.append(rl["kl_to_uniform"]); ys.append(a5["reject_rate"]); lab.append(f"{run}:{vid}")
    r = stats.spearmanr(xs, ys)
    out["kl_vs_power"] = dict(n_variants=len(xs), spearman_rho=float(r.statistic), p=float(r.pvalue),
                              kl_needed_for_9_canaries=math.log(20) / 9, points=[dict(variant=l, kl=x, reject=y) for l, x, y in zip(lab, xs, ys)])

    # 14. detection curves: fraction of deleted-subject processes rejected within n canaries (n = 1..9), pooled over all
    #     replicates of a method (runs x unlearning seeds), percentile bootstrap over (run, subject) clusters (B = 2000, seed 0)
    groups = {"planted": [("main", "qwen3b/planted"), ("key1", "qwen3b/planted"), ("key2", "qwen3b/planted")],
              "retrain": [("main", "qwen3b/retrain"), ("key1", "qwen3b/retrain"), ("key2", "qwen3b/retrain")],
              "npo": [("main", "qwen3b/npo"), ("main", "qwen3b/npo-s1"), ("main", "qwen3b/npo-s2"), ("key1", "qwen3b/npo"), ("key2", "qwen3b/npo")],
              "ga": [("main", "qwen3b/ga"), ("main", "qwen3b/ga-s1"), ("main", "qwen3b/ga-s2"), ("key1", "qwen3b/ga"), ("key2", "qwen3b/ga")]}
    for v in ["rmu", "graddiff", "adaptive50", "adaptive90"]:
        groups[v] = [("main", f"qwen3b/{v}"), ("main", f"qwen3b/{v}-s1"), ("main", f"qwen3b/{v}-s2")]
    groups["phantom (all Qwen-3B)"] = None
    rng = np.random.default_rng(0)
    dc = {}
    for name, members in groups.items():
        rows = []  # (cluster, tau or inf)
        if members is None:
            for run in RUNS:
                for vid in P[run]:
                    if vid.startswith("qwen3b/"):
                        rows += [((run, p["subject"]), p) for p in P[run][vid] if p["role"] == "phantom"]
        else:
            for run, vid in members:
                rows += [((run, p["subject"]), p) for p in P[run][vid] if p["role"] == "deleted"]
        nmax = max(len(p["logW"]) - 1 for _, p in rows)
        tau = []
        for _, p in rows:
            hit = [i for i, x in enumerate(p["logW"]) if x >= LOG20 - 1e-9]
            tau.append(hit[0] if hit else np.inf)
        tau = np.array(tau, float)
        cl = collections.defaultdict(list)
        for i, (c, _) in enumerate(rows):
            cl[c].append(i)
        C = list(cl.values())
        ns = np.arange(1, nmax + 1)
        curve = [(tau <= n).mean() for n in ns]
        idx = rng.integers(0, len(C), (2000, len(C)))
        cnt = np.array([[np.sum(tau[c] <= n) for c in C] for n in ns])  # (n, clusters)
        sz = np.array([len(c) for c in C])
        boot = cnt[:, idx].sum(2) / sz[idx].sum(1)[None, :]
        dc[name] = dict(n=ns.tolist(), reject_within_n=curve, ci95_lo=np.quantile(boot, .025, axis=1).tolist(),
                        ci95_hi=np.quantile(boot, .975, axis=1).tolist(), n_processes=len(rows), n_clusters=len(C),
                        members=[f"{a}:{b}" for a, b in members] if members else "phantom processes of every Qwen-3B variant, 3 runs")
    out["detection_curves"] = dc

    # 15. empirical calibration: P(sup_n W_n >= 1/alpha) on null processes over an alpha grid (trained variants; phantoms of
    #     every variant + deleted subjects of base/retrain), all orderings and ordering 0, Clopper-Pearson CIs
    grid = load_json(os.path.join(RESULTS, "calibration_mc.json"))["alpha_grid"]
    nulls = []
    for run in RUNS:
        for vid in P[run]:
            v = vid.split("/")[1].split("-s")[0]
            for p in P[run][vid]:
                if p["role"] == "phantom" or (p["role"] == "deleted" and v in ("retrain", "base")):
                    nulls.append(p)
    cal = {}
    for name, ps in [("all_orderings", nulls), ("ordering0", [p for p in nulls if p["order"] == 0])]:
        sup = np.array([max(p["logW"]) for p in ps])
        ks = [int(np.sum(sup >= math.log(1 / a) - 1e-9)) for a in grid]
        cis = [E.clopper_pearson(k, len(ps)) for k in ks]
        cal[name] = dict(n=len(ps), k=ks, rate=[k / len(ps) for k in ks], ci95=cis,
                         max_rate_over_alpha=float(max(k / len(ps) / a for k, a in zip(ks, grid))),
                         alphas_with_cp_lower_above_alpha=[a for c, a in zip(cis, grid) if c[0] > a])
    cal["alpha_grid"] = grid
    out["calibration_empirical"] = cal

    # 16. baseline comparison per behaviour group (same replicate pools as the detection curves): rejection rate of each test
    #     over all processes, percentile bootstrap over (run, subject) clusters
    tests = ["ours", "binom10_fixedn@0.05", "twin_eproc@0.05", "twin_ppr@0.05", "twin_fixedn@0.05", "binom10_peek@0.05"]
    bp = {}
    for name, members in groups.items():
        if members is None:
            continue
        rows = [((run, p["subject"]), p) for run, vid in members for p in P[run][vid] if p["role"] == "deleted"]
        cl = collections.defaultdict(list)
        for i, (c, _) in enumerate(rows):
            cl[c].append(i)
        C = list(cl.values())
        idx = rng.integers(0, len(C), (2000, len(C)))
        bp[name] = {}
        for t in tests:
            x = np.array([max(p["logW"]) >= LOG20 - 1e-9 if t == "ours" else p["base"][t] for _, p in rows], float)
            kc = np.array([x[c].sum() for c in C]); nc = np.array([len(c) for c in C])
            br = kc[idx].sum(1) / nc[idx].sum(1)
            bp[name][t] = dict(rate=float(x.mean()), k=int(x.sum()), n=len(x), ci95=[float(np.quantile(br, .025)), float(np.quantile(br, .975))])
        # canaries consumed: the e-process stops at its first crossing (else uses all n); fixed-n tests always use all n
        used = np.array([next((i for i, x in enumerate(p["logW"]) if x >= LOG20 - 1e-9), len(p["logW"]) - 1) for _, p in rows], float)
        nall = np.array([len(p["logW"]) - 1 for _, p in rows], float)
        bp[name]["canaries_used"] = dict(eprocess_mean=float(used.mean()), fixed_n=float(nall.mean()),
                                         eprocess_mean_among_rejected=float(used[used < nall].mean()) if np.any(used < nall) else None,
                                         saving_fraction=float(1 - used.mean() / nall.mean()))
    out["baseline_power_by_group"] = bp
    save_json(out, os.path.join(RESULTS, "stats_tests.json"))
    print("[stats] wrote results/stats_tests.json;", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
