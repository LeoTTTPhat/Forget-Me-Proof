"""Stage 8 (revision 2) analyses, CPU only -> results/revise2/<part>.json.

python src/revise2_analysis.py [part ...] [--force]
Parts:
  phantom9     fixed-order 9-canary phantom audits (k/140) and (key, subject)-cluster bootstrap intervals
  grinding2    key grinding with G = 2, 5, 10 keys (same procedure as results/revise1/grinding.json)
  oos2         out-of-sample rank-law predictions restricted to non-trivial variants; bias; heterogeneous model
  tertiles     memorisation tertiles: NPO vs compliant retrain within each tertile (null control)
  exact        exact DP: null size and within-budget threshold of the plug-in e-process; P(tau <= n) under
               pooled-i.i.d. and per-position rank laws for the committed bets; certificates (tightened Thm 2(v),
               fixed-bet Azuma bound, E[tau] bound from summed tails)
  recommended  kind mixture of fixed prior-shaped bets (KM-prior): lambda selected on key-0 Qwen2.5-3B, confirmed on
               the other replicates; null rates with cluster CIs, routing, canaries used, comparison with the default
  attribute    name-free genuine-attribute probes (results/rev2_attr): detection, any-name-token router, semantic filter,
               phantom nulls, 5-kind mixtures
  misc         small tests requested by reviewers (sampled-answer chance check, MIA conformal note)
"""
import argparse, collections, json, math, os, sys, time
import numpy as np
from scipy import stats as st
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import read_jsonl, load_json, save_json, stable_int
from fmp import eprocess as E
from fmp.analyze import ordering
import rev1lib as L
import revise1_analysis as R1

OUT = os.path.join(L.ROOT, "results", "revise2")
ALPHA = 0.05
M = 10
Q = E.geom_prior(M)
B_PRIOR = 2.0 * Q  # a0 * q_r


def save(part, obj):
    os.makedirs(OUT, exist_ok=True)
    save_json(obj, os.path.join(OUT, f"{part}.json"))
    print(f"[revise2] wrote {part}.json", flush=True)


def cl_boot(items, keyf, fld, B=4000, seed=0):
    g = collections.defaultdict(list)
    for p in items:
        g[keyf(p)].append(float(p[fld]))
    G = list(g.values()); kc = np.array([sum(x) for x in G]); nc = np.array([len(x) for x in G], float)
    idx = np.random.default_rng(seed).integers(0, len(G), (B, len(G)))
    r = kc[idx].sum(1) / nc[idx].sum(1)
    return dict(n_clusters=len(G), ci95=[float(np.quantile(r, .025)), float(np.quantile(r, .975))])


# ------------------------------------------------------------------------------------------------ phantom9
PH9 = [("key0", "qwen3b__base", "rev1_ph9"), ("key0", "qwen3b__retrain", "rev1_ph9"), ("key1", "qwen3b__retrain", "rev1_ph9_key1"),
       ("key2", "qwen3b__retrain", "rev1_ph9_key2"), ("key0", "llama3b__retrain", "rev1_ph9"),
       ("key0", "qwen3b__planted", "rev1_ph9"), ("key0", "qwen3b__npo", "rev1_ph9")]


def ph9_variant(key, name, sub):
    extra = [r for r in read_jsonl(os.path.join(L.ROOT, "results", sub, "probes", f"{name}.jsonl")) if r["role"] == "phantom"]
    base = [r for r in read_jsonl(os.path.join(L.ROOT, L.RUNDIR[key], "probes", f"{name}.jsonl")) if r["role"] == "phantom"]
    Vx = R1.V(key, name + "@ph9", rows=base + extra)
    Vx.vid = name.replace("__", "/")  # same tie-break seeds as results/revise1/typeI2.json
    return Vx


def part_phantom9():
    procs = []
    for key, name, sub in PH9:
        Vx = ph9_variant(key, name, sub)
        for a in Vx.subjects("phantom"):
            for o in range(3):
                W = R1.wealth_default(Vx, a, o)
                procs.append(dict(key=key, variant=name, subject=a, order=o, acc=L.first_cross(W, ALPHA) is not None))
    res = {}
    for nm, ps in (("all_orderings", procs), ("ordering0", [p for p in procs if p["order"] == 0])):
        k = sum(p["acc"] for p in ps)
        res[nm] = dict(L.rate(k, len(ps)), cluster_key_subject=cl_boot(ps, lambda p: (p["key"], p["subject"]), "acc"))
    res["per_variant_order0"] = {f"{p0}/{p1}": L.rate(sum(p["acc"] for p in procs if p["order"] == 0 and p["key"] == p0 and p["variant"] == p1),
                                                       sum(1 for p in procs if p["order"] == 0 and p["key"] == p0 and p["variant"] == p1))
                                 for p0, p1, _ in PH9}
    res["_notes"] = "Clusters = (key, phantom subject): the same 20 phantoms recur on every model of a key and the 3 orders reuse ranks."
    save("phantom9", res)


# ------------------------------------------------------------------------------------------------ grinding2
def part_grinding2():
    Vb = R1.V("key0", "qwen3b__base")
    rng = np.random.default_rng(11)
    res = {}
    for target in ["qwen3b__retrain", "llama3b__retrain"]:
        Vt = R1.V("key0", target)
        subs = [a for a in Vb.subjects("deleted") if a in Vt.subjects("deleted")]

        def rank_table(Vx, a):
            T = []
            for s in R1.ord0(Vx, a):
                sc = L.zagg(Vx.X[(a, s)]["S"])
                T.append([L.tie_rank(sc, p, stable_int("grind", Vx.vid, a, s, p)) for p in range(10)])
            return np.array(T)
        tabs = {a: (rank_table(Vb, a), rank_table(Vt, a)) for a in subs}
        out = {}
        for G in [1, 2, 5, 10]:
            acc = []
            for rep in range(50):
                for a in subs:
                    Rb, Rt = tabs[a]
                    P = rng.integers(0, 10, size=(G, Rb.shape[0]))
                    Wb = E.eprocess_paths_vec(Rb[np.arange(Rb.shape[0])[None, :], P], 10, "plugin")
                    best = int(np.argmax(Wb.max(1)))
                    rt = Rt[np.arange(Rt.shape[0]), P[best]]
                    acc.append(L.first_cross(E.eprocess_path(list(rt), 10, "plugin"), ALPHA) is not None)
            out[str(G)] = dict(false_accusation_target=float(np.mean(acc)), n=len(acc),
                               ci95=E.clopper_pearson(int(np.sum(acc)), len(acc)))
        res[target] = out
    res["_notes"] = ("Same procedure as results/revise1/grinding.json (G keys tried against the never-trained Qwen2.5-3B, best kept, "
                     "compliant retrain audited) with 50 instead of 20 repetitions and a different random stream; G = 1 is the honest rate.")
    save("grinding2", res)


# ------------------------------------------------------------------------------------------------ oos2
def part_oos2():
    t = load_json(os.path.join(L.ROOT, "results/revise1/thm2.json"))
    rows = t["per_variant"]
    def summ(rr):
        o = np.array([r["oos_observed"] for r in rr]); pi = np.array([r["oos_pred_iid"] for r in rr]); ph = np.array([r["oos_pred_heterogeneous"] for r in rr])
        return dict(n=len(rr), mae_iid=float(np.abs(o - pi).mean()), bias_iid_pred_minus_obs=float((pi - o).mean()),
                    max_under_iid=float((o - pi).max()), max_over_iid=float((pi - o).max()),
                    mae_het=float(np.abs(o - ph).mean()), bias_het=float((ph - o).mean()),
                    pearson_iid=float(np.corrcoef(o, pi)[0, 1]) if len(rr) > 2 else None,
                    pearson_het=float(np.corrcoef(o, ph)[0, 1]) if len(rr) > 2 else None)
    nonnull = [r for r in rows if r["group"] not in ("compliant", "never_trained")]
    nontriv = [r for r in rows if max(r["oos_observed"], r["oos_pred_iid"]) >= 0.1]
    res = dict(all=summ(rows), nonnull=summ(nonnull), nontrivial_obs_or_pred_ge_0_1=summ(nontriv),
               worst_over=max(rows, key=lambda r: r["oos_pred_iid"] - r["oos_observed"])["variant"],
               _notes="From results/revise1/thm2.json per_variant (half-split out-of-sample; heterogeneous = 3 position groups).")
    ww = max(rows, key=lambda r: r["oos_pred_iid"] - r["oos_observed"])
    res["worst_over_detail"] = dict(key=ww["key"], variant=ww["variant"], pred=ww["oos_pred_iid"], obs=ww["oos_observed"])
    save("oos2", res)


# ------------------------------------------------------------------------------------------------ tertiles
def part_tertiles():
    s = load_json(os.path.join(L.ROOT, "results/stats_tests.json"))["memorisation_bucket_trend_top1"]
    res = {}
    for m, alt in (("qwen3b", "npo"), ("llama3b", "npo"), ("qwen3b", "planted"), ("llama3b", "planted"), ("qwen3b", "rmu")):
        a, c = s[f"{m}/{alt}"], s[f"{m}/retrain"]
        rows = []
        for i in range(3):
            ka, na = round(a["props"][i] * a["n"][i]), a["n"][i]; kc, nc = round(c["props"][i] * c["n"][i]), c["n"][i]
            p = st.fisher_exact([[ka, na - ka], [kc, nc - kc]], alternative="greater")[1]
            rows.append(dict(tertile=i + 1, alt_top1=ka / na, compliant_top1=kc / nc, diff=ka / na - kc / nc, fisher_p_greater=float(p)))
        res[f"{m}/{alt}_vs_retrain"] = dict(rows=rows, trend_p_alt=a["p"], trend_p_compliant=c["p"])
    res["_notes"] = ("Tertiles of the lazy model's pre-deletion rank of the planted value (60 canaries each, 20 subjects x 3 keys "
                     "for Qwen; key 0 x 3 orderings-free canary count for Llama as in stats_tests.json). The compliant retrain shows a "
                     "trend too (selection of canaries whose planted value the base model already prefers); the within-tertile "
                     "difference to the compliant model is the null-controlled quantity. Canaries of a subject are not independent, so "
                     "Fisher p-values are approximate.")
    save("tertiles", res)


# ------------------------------------------------------------------------------------------------ exact DP
def logW_plugin(cnt):
    n = sum(cnt)
    return n * math.log(M) + float(sum(gammaln(c + b) - gammaln(b) for c, b in zip(cnt, B_PRIOR))) + gammaln(2.0) - gammaln(n + 2.0)


def make_logW_fixed(f):
    lf = np.log(f)
    return lambda cnt: float(np.dot(cnt, lf))


def dp_cross(laws, logW, c):
    """Exact P(tau <= n), n = 1..len(laws), for a bet whose wealth depends only on the rank counts; threshold log-wealth c."""
    alive = {tuple([0] * M): 1.0}
    cum, out = 0.0, []
    for P in laws:
        nxt = collections.defaultdict(float)
        for cnt, pr in alive.items():
            for r in range(M):
                if P[r] > 0:
                    k = list(cnt); k[r] += 1
                    nxt[tuple(k)] += pr * P[r]
        alive = {}
        for cnt, pr in nxt.items():
            if logW(cnt) >= c - 1e-12:
                cum += pr
            else:
                alive[cnt] = pr
        out.append(cum)
    return out


def null_threshold(logW, n=9, alpha=ALPHA):
    U = [np.full(M, 1 / M)] * n
    lo, hi = 0.0, math.log(1 / alpha)
    if dp_cross(U, logW, hi)[-1] > alpha:
        raise RuntimeError("Ville bound violated?")
    for _ in range(40):
        mid = (lo + hi) / 2
        if dp_cross(U, logW, mid)[-1] <= alpha:
            hi = mid
        else:
            lo = mid
    return hi, dp_cross(U, logW, hi)[-1]


def rank_laws(Vx):
    subs = Vx.subjects("deleted")
    R = np.array([L.subject_rank_seq(Vx, a, 0)[0] for a in subs])
    pooled = np.bincount(R.ravel(), minlength=M + 1)[1:] + 0.5; pooled = pooled / pooled.sum()
    groups = [range(0, 5), range(5, 7), range(7, 9)]
    glaw = []
    for g in groups:
        c = np.bincount(R[:, list(g)].ravel(), minlength=M + 1)[1:] + 0.1; glaw.append(c / c.sum())
    grouped = [glaw[0] if j < 5 else glaw[1] if j < 7 else glaw[2] for j in range(9)]
    perpos = []
    for j in range(9):
        c = np.bincount(R[:, j], minlength=M + 1)[1:] + 2.0 * pooled; perpos.append(c / c.sum())  # shrink to pooled (2 pseudo-obs)
    return R, pooled, grouped, perpos


def regret_bound(n, tight=True):
    lb = np.maximum(0.0, np.log(1 / B_PRIOR))
    if not tight:
        return (M + 3) / 2 * math.log(n + 1) + 1 / 12 + float(lb.sum())
    k = min(n, M)
    return (k + 3) / 2 * math.log(n + 1) + 1 / 12 + float(np.sort(lb)[::-1][:k].sum())


def azuma_B(P, gamma_c_fn, beta=0.1, alpha=ALPHA, regret=lambda n: 0.0, nmax=20000):
    """Smallest n with exp(-2 s_n^2 / (n c^2)) <= beta, s_n = n gamma - regret(n) - log(1/alpha); and the bound
    E[tau] <= sum_{n>=0} min(1, tail_n) (tail_n = 1 when s_n <= 0)."""
    best = None
    for eta in np.linspace(0.0, 0.9, 19):
        gam, cc = gamma_c_fn(eta)
        if gam <= 0:
            continue
        B, Et = None, 0.0
        for n in range(0, nmax):
            s_ = n * gam - regret(n) - math.log(1 / alpha) if n > 0 else -1
            tail = 1.0 if s_ <= 0 else min(1.0, math.exp(-2 * s_ ** 2 / (n * cc ** 2)))
            Et += tail
            if B is None and s_ > 0 and tail <= beta:
                B = n
            if B is not None and tail < 1e-12:
                break
        if best is None or (B is not None and (best["B"] is None or B < best["B"])):
            best = dict(B=B, E_tau_bound=Et, eta=float(eta), gamma=gam, c=cc)
    return best


def part_exact():
    res = {}
    lw = logW_plugin
    c_star, size_star = null_threshold(lw)
    res["plugin_null"] = dict(size_at_1_over_alpha=dp_cross([np.full(M, .1)] * 9, lw, math.log(20))[-1],
                              size_curve=dp_cross([np.full(M, .1)] * 9, lw, math.log(20)),
                              exact_threshold_logW=c_star, exact_threshold_W=math.exp(c_star), exact_size=size_star)
    lams = (0.5, 0.75, 1.0)
    fixed = {lam: (1 - lam) + lam * M * Q for lam in lams}
    for lam, f in fixed.items():
        lwf = make_logW_fixed(f)
        cf, sf = null_threshold(lwf)
        res[f"prior_bet_{lam}_null"] = dict(size_at_1_over_alpha=dp_cross([np.full(M, .1)] * 9, lwf, math.log(20))[-1],
                                            exact_threshold_W=math.exp(cf), exact_size=sf)
    comp = {}
    for key, name in [("key0", "qwen3b__planted"), ("key0", "qwen3b__npo"), ("key0", "llama3b__planted"), ("key0", "qwen7b__planted"),
                      ("key0", "qwen3b__rmu"), ("key1", "qwen3b__planted"), ("key2", "qwen3b__planted")]:
        Vx = R1.V(key, name)
        R, pooled, grouped, perpos = rank_laws(Vx)
        obs = [float(np.mean([L.first_cross(E.eprocess_path(list(r), M, "plugin")[:n + 1], ALPHA) is not None for r in R])) for n in range(1, 10)]
        d = dict(observed_plugin=obs,
                 dp_plugin_iid=dp_cross([pooled] * 9, lw, math.log(20)),
                 dp_plugin_grouped=dp_cross(grouped, lw, math.log(20)),
                 dp_plugin_perpos=dp_cross(perpos, lw, math.log(20)),
                 dp_plugin_exactthr_grouped=dp_cross(grouped, lw, c_star))
        for lam, f in fixed.items():
            lwf = make_logW_fixed(f)
            d[f"observed_prior_{lam}"] = [float(np.mean([L.first_cross(np.r_[1.0, np.cumprod(f[np.array(r) - 1])][:n + 1], ALPHA) is not None for r in R]))
                                          for n in range(1, 10)]
            d[f"dp_prior_{lam}_grouped"] = dp_cross(grouped, lwf, math.log(20))
        D = E.kl(pooled, np.full(M, .1))

        def gc_plugin(eta):
            Qx = (1 - eta) * pooled + eta / M
            return float((pooled * np.log(M * Qx)).sum()), float(np.log(Qx.max() / Qx.min()))
        d["D"] = float(D)
        d["thm2v_original"] = azuma_B(pooled, gc_plugin, regret=lambda n: regret_bound(n, tight=False))
        d["thm2v_tightened"] = azuma_B(pooled, gc_plugin, regret=lambda n: regret_bound(n, tight=True))
        for lam, f in fixed.items():
            g_ = float((pooled * np.log(f)).sum()); c_ = float(np.log(f.max() / f.min()))
            d[f"fixed_prior_{lam}_azuma"] = azuma_B(pooled, lambda eta, g_=g_, c_=c_: (g_, c_))
        d["regret_bound_at_9_original"] = regret_bound(9, False); d["regret_bound_at_9_tight"] = regret_bound(9, True)
        # actual regret of the plug-in bet on the observed sequences: n KL(P_hat_n || U) - log W_n at n = 9
        act = []
        for r in R:
            cnt = np.bincount(r, minlength=M + 1)[1:]; ph = cnt / cnt.sum()
            nkl = float(9 * np.sum([p * math.log(M * p) for p in ph if p > 0]))
            act.append(nkl - logW_plugin(tuple(cnt)))
        d["actual_regret_at_9"] = dict(mean=float(np.mean(act)), max=float(np.max(act)), min=float(np.min(act)))
        comp[f"{key}/{name}"] = d
    res["completeness"] = comp
    res["_notes"] = ("Exact dynamic programming over rank-count vectors (the wealth of the plug-in and fixed bets depends only on the "
                     "counts). grouped = per-position laws for positions 1-5 / 6-7 / 8-9 (planting rates 4 / 2 / 1) estimated in sample "
                     "(add-0.1); perpos = each position's 20 ranks shrunk towards the pooled law; iid = pooled law (add-0.5). "
                     "thm2v_tightened uses |S_n| <= min(n, M) and the observed-support prior term with max(0, log 1/b_r). "
                     "fixed_prior_*_azuma: Azuma bound for a non-learning bet (no regret term), c = log(max f / min f). E_tau_bound = "
                     "sum_n min(1, tail bound) (Thm 2(vi)). exact_threshold_W: smallest wealth threshold with exact null size <= alpha "
                     "within a 9-canary budget (valid only for that fixed budget).")
    save("exact", res)


# ------------------------------------------------------------------------------------------------ recommended verifier
def km_prior_path(kseq, lam, kinds=L.KINDS):
    f = (1 - lam) + lam * M * Q
    Wk = {k: 1.0 for k in kinds}
    W = [1.0]
    for kr in kseq:
        for k in kinds:
            Wk[k] *= f[kr[k] - 1]
        W.append(float(np.mean([Wk[k] for k in kinds])))
    return np.array(W)


def km_plugin_path(kseq, kinds=L.KINDS):
    bets = {k: E.PluginBet(M) for k in kinds}
    Wk = {k: 1.0 for k in kinds}
    W = [1.0]
    for kr in kseq:
        for k in kinds:
            Wk[k] *= bets[k].e(kr[k]); bets[k].update(kr[k])
        W.append(float(np.mean([Wk[k] for k in kinds])))
    return np.array(W)


LAMS = (0.5, 0.75, 1.0)


_WSTAR = []


def wstar():
    if not _WSTAR:
        c, _ = null_threshold(logW_plugin)
        _WSTAR.append(math.exp(c))
    return _WSTAR[0]


def verifiers(Vx, a, order=0):
    """first-crossing times of the verifiers compared in revision 2 (alpha = 0.05)."""
    kseq = R1.kind_seq(Vx, a, order)
    rk, _ = L.subject_rank_seq(Vx, a, order)
    Wd = E.eprocess_path(rk, M, "plugin")
    out = dict(default=L.first_cross(Wd, ALPHA),
               kind_mix=L.first_cross(km_plugin_path(kseq), ALPHA))
    hit = [n for n in range(1, min(len(Wd), 10)) if Wd[n] >= wstar() - 1e-9]
    out["plugin_exact9"] = hit[0] if hit else None  # within-budget exact threshold (valid for the fixed 9-canary budget only)
    for lam in LAMS:
        out[f"km_prior_{lam}"] = L.first_cross(km_prior_path(kseq, lam), ALPHA)
    out["n"] = len(rk)
    return out


def part_recommended():
    rows = []
    for key, name in R1.ALLTRAINED:
        Vx = R1.V(key, name)
        null_v = (key, name) in R1.COMPLIANT
        for role in ("deleted", "phantom"):
            for a in Vx.subjects(role):
                for o in range(3):
                    r = verifiers(Vx, a, o)
                    r.update(key=key, variant=name, role=role, subject=a, order=o, group=R1.group_of(name),
                             null=(role == "phantom" or null_v))
                    rows.append(r)
    # 9-canary phantom audits
    for key, name, sub in PH9:
        Vx = ph9_variant(key, name, sub)
        for a in Vx.subjects("phantom"):
            for o in range(3):
                r = verifiers(Vx, a, o)
                r.update(key=key, variant=name + "@ph9", role="phantom9", subject=a, order=o, group="phantom9", null=True)
                rows.append(r)
    V_ = ["default", "kind_mix", "plugin_exact9"] + [f"km_prior_{l}" for l in LAMS]
    sel = lambda r: r["key"] == "key0" and r["variant"].startswith("qwen3b")
    res = {}
    pw = lambda rr, v: L.rate(sum(r[v] is not None for r in rr), len(rr))
    nn0 = [r for r in rows if r["role"] == "deleted" and not r["null"] and r["order"] == 0]
    res["selection_key0_qwen"] = {v: pw([r for r in nn0 if sel(r)], v) for v in V_}
    res["confirmation_other"] = {v: pw([r for r in nn0 if not sel(r)], v) for v in V_}
    res["power_all_nonnull_order0"] = {v: pw(nn0, v) for v in V_}
    res["power_by_group_order0"] = {g: {v: pw([r for r in nn0 if r["group"] == g], v) for v in V_}
                                    for g in sorted({r["group"] for r in nn0})}
    best = max(LAMS, key=lambda l: (res["selection_key0_qwen"][f"km_prior_{l}"]["k"], -l))
    res["selected_lambda_on_key0"] = best
    nul = [r for r in rows if r["null"] and r["role"] != "phantom9"]
    res["null_all_orderings"] = {v: dict(pw(nul, v), cluster=cl_boot([dict(r, acc=r[v] is not None) for r in nul],
                                                                     lambda r: (r["key"], r["subject"], r["role"]), "acc")) for v in V_}
    res["null_ordering0"] = {v: pw([r for r in nul if r["order"] == 0], v) for v in V_}
    p9 = [r for r in rows if r["role"] == "phantom9"]
    res["null_phantom9"] = {v: dict(pw(p9, v), order0=pw([r for r in p9 if r["order"] == 0], v),
                                    cluster=cl_boot([dict(r, acc=r[v] is not None) for r in p9], lambda r: (r["key"], r["subject"]), "acc")) for v in V_}
    res["canaries_used_order0"] = {g: {v: float(np.mean([r[v] or 9 for r in nn0 if r["group"] == g and r["n"] == 9])) for v in V_}
                                   for g in ("lazy", "npo", "rmu")}
    comp0 = [r for r in rows if r["role"] == "deleted" and r["group"] == "compliant" and r["order"] == 0]
    res["canaries_used_order0"]["compliant"] = {v: float(np.mean([r[v] or 9 for r in comp0])) for v in V_}
    # paired comparison with the default (cluster sign-flip permutation over (key, subject))
    def perm(t1, t2, B=20000):
        cl = collections.defaultdict(float)
        for r in nn0:
            cl[(r["key"], r["subject"])] += (r[t1] is not None) - (r[t2] is not None)
        d = np.array(list(cl.values())); obs = d.sum()
        sims = (np.random.default_rng(5).choice([-1, 1], size=(B, len(d))) * d).sum(1)
        return dict(diff=float(obs), p_two_sided=float((np.abs(sims) >= abs(obs) - 1e-9).mean()))
    res["perm_vs_default"] = {v: perm(v, "default") for v in V_ if v != "default"}
    # routing / refusal robustness (ordering 0), same derived providers as results/revise1/routing.json
    rt = {}
    for key, tag, routes in [("key0", "qwen3b", ["retrain", "base"]), ("key0", "llama3b", ["retrain"]), ("key1", "qwen3b", ["retrain"]),
                             ("key2", "qwen3b", ["retrain"])]:
        beh = {"lazy": R1.V(key, f"{tag}__planted")}
        for rule in ["kw_full", "semantic"]:
            beh[f"refuse_{rule}"] = R1.V(key, f"{tag}__filter_{rule}", rows=R1.derived_rows(key, tag, rule))
            for r_ in routes:
                beh[f"route_{r_}_{rule}"] = R1.V(key, f"{tag}__route_{rule}" + ("" if r_ == "retrain" else f"_{r_}"),
                                                 rows=R1.derived_rows(key, tag, rule, route_from=f"{tag}__{r_}"))
        rt[f"{key}/{tag}"] = {b: {v: pw([dict(verifiers(Vx, a, 0)) for a in Vx.subjects("deleted")], v) for v in V_} for b, Vx in beh.items()}
    res["routing_order0"] = rt
    # scrubbing providers (results/rev1)
    sc = {}
    for nm in ("scrub-same", "scrub-heldout"):
        f = os.path.join(L.ROOT, "results/rev1/probes", f"qwen3b__{nm}.jsonl")
        if os.path.exists(f):
            Vx = R1.V("key0", f"qwen3b__{nm}", rows=read_jsonl(f))
            sc[nm] = {v: pw([verifiers(Vx, a, 0) for a in Vx.subjects("deleted")], v) for v in V_}
    res["scrubbing_order0"] = sc
    res["_notes"] = ("KM-prior = average over the 4 probe kinds of fixed prior-shaped e-processes prod_j f(R^t_j), f = (1-lam) + lam M q "
                     "(q geometric prior of Eq. 2): valid by Prop. 2, no learning. lam selected by detection on key-0 Qwen2.5-3B "
                     "(ties -> smaller lam), confirmed on the other 11 replicates. kind_mix = Eq. (3) with plug-in bets.")
    save("recommended", res)
    save_json(rows, os.path.join(OUT, "recommended_rows.json"), indent=None)


# ------------------------------------------------------------------------------------------------ attribute probes
ATTR_VARIANTS = [("key0", "qwen3b__planted", "lazy"), ("key0", "qwen3b__retrain", "compliant"), ("key0", "qwen3b__base", "never_trained"),
                 ("key0", "qwen3b__npo", "npo"), ("key0", "llama3b__planted", "lazy"), ("key0", "llama3b__retrain", "compliant"),
                 ("key0", "qwen3b__scrub-same", "scrub")]
KINDS5 = L.KINDS + ["attribute"]


def attr_rank(Va, a, s, tag="attr"):
    e = Va.X[(a, s)]
    return L.tie_rank(L.zagg(e["S"]), e["p"], stable_int("tie", Va.vid, a, s, M, tag))


def part_attribute():
    feats_a = load_json(os.path.join(L.ROOT, "results/rev2_attr/filters/probe_filter_feats.json"))
    feats_m = load_json(os.path.join(L.ROOT, "results/main/filters/probe_filter_feats.json"))
    raw_pl = read_jsonl(os.path.join(L.ROOT, "results/main/probes/qwen3b__planted.jsonl"))
    ret = [r["key"] for r in raw_pl if r["role"] == "retained" and r["key"] in feats_m]
    tau = float(np.quantile([feats_m[k]["sem"] for k in ret], 0.95))
    rows_a = {}
    res = dict(semantic_tau=tau)
    # filter catch rates on attribute probes
    keys_by_role = collections.defaultdict(list)
    for r in read_jsonl(os.path.join(L.ROOT, "results/rev2_attr/probes/qwen3b__planted.jsonl")):
        keys_by_role[r["role"]].append(r["key"])
    res["filter_catch_attribute_probes"] = {role: dict(kw_full=float(np.mean([feats_a[k]["kw_full"] for k in ks])),
                                                       kw_tok=float(np.mean([feats_a[k]["kw_tok"] for k in ks])),
                                                       semantic=float(np.mean([feats_a[k]["sem"] >= tau for k in ks])))
                                            for role, ks in keys_by_role.items()}
    res["filter_catch_main_probes"] = {role: dict(kw_tok=float(np.mean([feats_m[r["key"]]["kw_tok"] for r in raw_pl if r["role"] == role])),
                                                  semantic=float(np.mean([feats_m[r["key"]]["sem"] >= tau for r in raw_pl if r["role"] == role])))
                                       for role in ("deleted", "retained")}
    det = {}
    for key, name, grp in ATTR_VARIANTS:
        f = os.path.join(L.ROOT, "results/rev2_attr/probes", f"{name}.jsonl")
        if not os.path.exists(f):
            det[name] = "missing"; continue
        Va = L.load_variant(key, name, rows=read_jsonl(f)); Va.vid = name.replace("__", "/") + "@attr"
        mf = os.path.join(L.ROOT, "results/rev1/probes" if "scrub" in name else L.RUNDIR[key] + "/probes", f"{name}.jsonl")
        Vm = R1.V(key, name, rows=read_jsonl(mf)) if "scrub" in name else R1.V(key, name)
        d = {}
        for role in ("deleted", "retained", "phantom"):
            subs = [a for a in Va.subjects(role)]
            if not subs:
                continue
            acc = collections.defaultdict(list); uni = []
            for a in subs:
                for o in range(3 if role == "phantom" else 1):
                    sl = ordering(Va.slots(a), a, o)
                    ra = [attr_rank(Va, a, s) for s in sl]
                    uni += ra if o == 0 else []
                    acc["attribute_only"].append(L.first_cross(E.eprocess_path(ra, M, "plugin"), ALPHA) is not None)
                    acc["attribute_only_prior0.5"].append(L.first_cross(np.r_[1.0, np.cumprod(((0.5) + 0.5 * M * Q)[np.array(ra) - 1])], ALPHA) is not None)
                    if role == "phantom" or a not in Vm.subjects(role) or len(Vm.slots(a)) != len(sl):
                        continue
                    ks = [dict(L.kind_ranks(Vm, a, s), attribute=r_) for s, r_ in zip(sl, ra)]
                    acc["kind_mix5"].append(L.first_cross(km_plugin_path(ks, KINDS5), ALPHA) is not None)
                    acc["km_prior5_0.5"].append(L.first_cross(km_prior_path(ks, 0.5, KINDS5), ALPHA) is not None)
                    acc["kind_mix4"].append(L.first_cross(km_plugin_path(ks, L.KINDS), ALPHA) is not None)
                    # any-name-token router: every main probe is caught; refused (random ranks) or routed to the compliant model
                    if role == "deleted":
                        rk_ref = [dict({k: L.tie_rank(np.zeros(M), Vm.X[(a, s)]["p"], stable_int("ref", a, s, k)) for k in L.KINDS}, attribute=r_)
                                  for s, r_ in zip(sl, ra)]
                        acc["anytok_refuse_kind_mix5"].append(L.first_cross(km_plugin_path(rk_ref, KINDS5), ALPHA) is not None)
                        acc["anytok_refuse_km_prior5"].append(L.first_cross(km_prior_path(rk_ref, 0.5, KINDS5), ALPHA) is not None)
                        if key == "key0" and name.endswith("__planted"):
                            tag = name.split("__")[0]
                            Vc = R1.V(key, f"{tag}__retrain")
                            if a in Vc.subjects("deleted"):
                                rk_rt = [dict(L.kind_ranks(Vc, a, s), attribute=r_) for s, r_ in zip(sl, ra)]
                                acc["anytok_route_kind_mix5"].append(L.first_cross(km_plugin_path(rk_rt, KINDS5), ALPHA) is not None)
                                acc["anytok_route_km_prior5"].append(L.first_cross(km_prior_path(rk_rt, 0.5, KINDS5), ALPHA) is not None)
            cnt = np.bincount(uni, minlength=M + 1)[1:]
            d[role] = {k: L.rate(sum(v), len(v)) for k, v in acc.items()}
            d[role]["top1_attribute"] = float(np.mean(np.array(uni) == 1)); d[role]["chi2_p"] = float(st.chisquare(cnt).pvalue)
            d[role]["n_canaries"] = len(uni)
        det[name] = dict(group=grp, **d)
    res["detection"] = det
    res["_notes"] = ("Attribute probes (4 per canary, kind 'attribute'): the subject is identified only by a genuine TOFU descriptor or "
                     "book title, never by a name token (src/make_attr_refs.py). Ordering 0 for deleted/retained, 3 orderings for the "
                     "9-canary phantoms. anytok_refuse: all 16 name-bearing probes refused (constant rows -> uniform random ranks), "
                     "attribute probes answered by the provider's model; anytok_route: name-bearing probes answered by the compliant retrain. "
                     "kind_mix5 / km_prior5 add the attribute kind to Eq. (3) (plug-in / fixed prior bet lam=0.5); these 5-kind verifiers "
                     "were defined after the 16-probe experiments and are reported as a probe-design extension.")
    save("attribute", res)


# ------------------------------------------------------------------------------------------------ misc
def part_misc():
    res = {}
    # compliant top-1 0.25 at T = 0.7 (5 of 20 canaries; chance 0.1)
    s2 = load_json(os.path.join(L.ROOT, "results/revise1/sampled2.json"))
    res["sampled_compliant_T0.7"] = dict(k=5, n=20, p_one_sided=float(st.binom.sf(4, 20, 0.1)),
                                         note="binomial(20, 0.1) upper tail; 3 temperatures were inspected on the one compliant variant")
    res["sampled2_keys"] = list(s2.keys())[:20]
    res["mia_conformal"] = ("Leave-one-out conformal p-values among 20 exchangeable phantoms are a permutation of {1/20,...,20/20}, "
                            "so exactly one phantom has p = 1/20 <= 0.05 in every row: the 1/20 phantom rate holds by construction "
                            "and is the exact level, not evidence of validity; validity rests on exchangeability of phantom and deleted "
                            "subjects under the null.")
    save("misc", res)


PARTS = ["phantom9", "grinding2", "oos2", "tertiles", "exact", "recommended", "attribute", "misc"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in a.parts or PARTS:
        if not a.force and os.path.exists(os.path.join(OUT, f"{p}.json")):
            print(f"[revise2] skip {p} (exists)"); continue
        t = time.time()
        globals()[f"part_{p}"]()
        print(f"[revise2] {p} took {time.time() - t:.0f}s", flush=True)
