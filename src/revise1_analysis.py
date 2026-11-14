"""Stage 7 (revision 1) re-analyses of the existing probe logs (CPU only) -> results/revise1/<part>.json.

python src/revise1_analysis.py [part ...]      (default: all parts; a part is skipped when its JSON exists, use --force)
Parts: routing audit_route config_select grinding baselines2 mia2 typeI2 thm2 filters_paired unseen heldout sampled2
       small_alpha newvariants
Every number added to the paper in revision 1 comes from these files (see paper/RESPONSE.md).
"""
import argparse, collections, itertools, json, math, os, sys, time
import numpy as np
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import read_jsonl, load_json, save_json, stable_int
from fmp import eprocess as E, canary as C
from fmp.analyze import ordering, mia_baseline
import rev1lib as L

OUT = os.path.join(L.ROOT, "results", "revise1")
ALPHA = 0.05
_V = {}


def V(key, name, sub="probes", rows=None, cache=True):
    k = (key, name, sub)
    if rows is not None:
        return L.load_variant(key, name, sub, rows)
    if k not in _V:
        _V[k] = L.load_variant(key, name, sub)
    return _V[k]


QWEN_K0 = ["planted", "retrain", "base", "npo", "npo-s1", "npo-s2", "ga", "ga-s1", "ga-s2", "graddiff", "graddiff-s1", "graddiff-s2",
           "rmu", "rmu-s1", "rmu-s2", "adaptive50", "adaptive50-s1", "adaptive50-s2", "adaptive90", "adaptive90-s1", "adaptive90-s2"]
NONNULL = ([("key0", f"qwen3b__{v}") for v in QWEN_K0 if v not in ("retrain", "base")] +
           [("key1", "qwen3b__planted"), ("key1", "qwen3b__npo"), ("key1", "qwen3b__ga"),
            ("key2", "qwen3b__planted"), ("key2", "qwen3b__npo"), ("key2", "qwen3b__ga"),
            ("key0", "llama3b__planted"), ("key0", "llama3b__npo"), ("key0", "llama3b__ga"),
            ("key0", "qwen7b__planted"), ("key0", "qwen7b__npo")])
COMPLIANT = [("key0", "qwen3b__retrain"), ("key1", "qwen3b__retrain"), ("key2", "qwen3b__retrain"), ("key0", "llama3b__retrain"),
             ("key0", "qwen3b__base")]
ALLTRAINED = NONNULL + COMPLIANT


def group_of(name):
    v = name.split("__")[1].split("-s")[0]
    return {"planted": "lazy", "retrain": "compliant", "base": "never_trained"}.get(v, v)


def save(part, obj):
    os.makedirs(OUT, exist_ok=True)
    save_json(obj, os.path.join(OUT, f"{part}.json"))
    print(f"[revise1] wrote {part}.json", flush=True)


def ord0(Vx, a):
    return ordering(Vx.slots(a), a, 0)


# ------------------------------------------------------------------------------------------------ verifiers on rank data
def wealth_default(Vx, a, order=0, pids=None, strategy="plugin", lam=0.5, alpha=ALPHA):
    rk, _ = L.subject_rank_seq(Vx, a, order, pids)
    return E.eprocess_path(rk, Vx.M, strategy, lam=lam)


def kind_seq(Vx, a, order=0):
    return [L.kind_ranks(Vx, a, s) for s in ordering(Vx.slots(a), a, order)]


AGGS = ["default", "namefree", "kind_avg", "kind_mix"]


def wealth_agg(Vx, a, agg, order=0):
    if agg == "default":
        return wealth_default(Vx, a, order)
    if agg == "namefree":
        return wealth_default(Vx, a, order, pids=L.NAME_FREE)
    if agg in ("kind_avg", "kind_mix"):
        return L.kindmix_path(kind_seq(Vx, a, order), Vx.M, "avg" if agg == "kind_avg" else "mix")
    raise ValueError(agg)


def count(Vx, role, agg="default", alpha=ALPHA, order=0):
    k = n = 0
    for a in Vx.subjects(role):
        W = wealth_agg(Vx, a, agg, order)
        k += L.first_cross(W, alpha) is not None; n += 1
    return k, n


# ------------------------------------------------------------------------------------------------ 1. routing / robust aggregation
def derived_rows(key, tag, rule, route_from=None, noise=False):
    """Rows of the planted log with the filtered probes refused (constant row), routed to another model's log, or
    replaced by pi-independent Gaussian noise."""
    feats = load_json(os.path.join(L.ROOT, "results/main/filters/probe_filter_feats.json"))
    raw = read_jsonl(os.path.join(L.ROOT, L.RUNDIR[key], "probes", f"{tag}__planted.jsonl"))
    if rule == "semantic":
        ret = [r["key"] for r in raw if r["role"] == "retained" and r["key"] in feats]
        tau = float(np.quantile([feats[k]["sem"] for k in ret], 0.95))
        f = lambda k: feats[k]["sem"] >= tau
    else:
        f = lambda k: feats[k][rule]
    src = {r["key"]: r for r in read_jsonl(os.path.join(L.ROOT, L.RUNDIR[key], "probes", f"{route_from}.jsonl"))} if route_from else {}
    out = []
    for r in raw:
        if r["key"] in feats and f(r["key"]):
            if route_from:
                # retained-subject probes are missing in some route-source logs; they are not used by the counts below
                r = dict(src[r["key"]]) if r["key"] in src else dict(r, refused=True)
            elif noise:
                g = np.random.default_rng(stable_int("noise", r["key"]) % 2 ** 32)
                r = dict(r, scores=[[float(x), 1] for x in g.normal(0, 10, len(r["scores"]))])
            else:
                r = dict(r, refused=True)
        out.append(r)
    return out


def part_routing():
    res = {}
    setups = [("key0", "qwen3b", ["retrain", "base"]), ("key0", "llama3b", ["retrain"]), ("key1", "qwen3b", ["retrain"]),
              ("key2", "qwen3b", ["retrain"])]
    for key, tag, routes in setups:
        beh = {"lazy": V(key, f"{tag}__planted"), "compliant": V(key, f"{tag}__retrain")}
        for rule in ["kw_full", "semantic", "kw_tok"]:
            beh[f"refuse_{rule}"] = V(key, f"{tag}__filter_{rule}", rows=derived_rows(key, tag, rule))
            for rt in routes:
                beh[f"route_{rt}_{rule}"] = V(key, f"{tag}__route_{rule}" + ("" if rt == "retrain" else f"_{rt}"),
                                              rows=derived_rows(key, tag, rule, route_from=f"{tag}__{rt}"))
            beh[f"noise_{rule}"] = V(key, f"{tag}__noise_{rule}", rows=derived_rows(key, tag, rule, noise=True))
        r = {}
        for b, Vx in beh.items():
            r[b] = {}
            for agg in AGGS:
                k, n = count(Vx, "deleted", agg)
                kp, np_ = count(Vx, "phantom", agg)
                r[b][agg] = dict(deleted=L.rate(k, n), phantom=L.rate(kp, np_))
        res[f"{key}/{tag}"] = r
    # power and validity of the robust aggregators on all trained variants (ordering 0)
    pw = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    nul = collections.defaultdict(lambda: [0, 0])
    for key, name in ALLTRAINED:
        Vx = V(key, name)
        for agg in AGGS:
            if (key, name) in NONNULL:
                k, n = count(Vx, "deleted", agg)
                pw[agg][group_of(name)][0] += k; pw[agg][group_of(name)][1] += n
                pw[agg]["all_nonnull"][0] += k; pw[agg]["all_nonnull"][1] += n
            else:
                k, n = count(Vx, "deleted", agg)
                nul[agg][0] += k; nul[agg][1] += n
            k, n = count(Vx, "phantom", agg)
            nul[agg][0] += k; nul[agg][1] += n
    res["power_nonnull_order0"] = {agg: {g: L.rate(*v) for g, v in d.items()} for agg, d in pw.items()}
    res["null_order0"] = {agg: L.rate(*v) for agg, v in nul.items()}
    res["_notes"] = ("Ordering 0. refuse = constant row; route_<m> = filtered probes answered by model m's logged scores "
                     "(base = never-trained Qwen2.5-3B, no retraining needed); noise = pi-independent N(0,10^2) scores. "
                     "Aggregators: default z-mean of 16 probes; namefree = z-mean of the 6 probes without the full name; "
                     "kind_avg = per-canary average of 4 kind-specific plug-in e-values; kind_mix = average of 4 kind-specific "
                     "plug-in e-processes. All four are valid (Lemma 1 per kind; averages of e-values / test martingales).")
    save("routing", res)


# ------------------------------------------------------------------------------------------------ 2. audit-then-route
def part_audit_route():
    res = {}
    for key, tag in [("key0", "qwen3b"), ("key1", "qwen3b"), ("key2", "qwen3b"), ("key0", "llama3b")]:
        Vl, Vc = V(key, f"{tag}__planted"), V(key, f"{tag}__retrain")
        r = {}
        for agg in ["default", "kind_mix"]:
            curve = []
            for t in range(0, 10):
                k = n = 0
                bound = []
                for a in Vl.subjects("deleted"):
                    sl = ord0(Vl, a)
                    if agg == "default":
                        rk = [L.canary_rank(Vl if i < t else Vc, a, s) for i, s in enumerate(sl)]
                        W = E.eprocess_path(rk, 10, "plugin")
                    else:
                        ks = [L.kind_ranks(Vl if i < t else Vc, a, s) for i, s in enumerate(sl)]
                        W = L.kindmix_path(ks, 10, "mix")
                    fc = L.first_cross(W, ALPHA)
                    k += fc is not None; n += 1
                    pre = L.first_cross(W[:t + 1], ALPHA)
                    bound.append(1.0 if pre is not None else min(1.0, ALPHA * W[t]))
                curve.append(dict(route_after=t, accused=L.rate(k, n), ville_bound_mean=float(np.mean(bound))))
            r[agg] = curve
        res[f"{key}/{tag}"] = r
    res["_notes"] = ("Audit-then-route: the provider answers the first t canaries of the ordering-0 audit with the planted (lazy) "
                     "model and every later probe with the compliant retrain. t=0 is a provider that recognises the auditor "
                     "immediately (passes like a compliant one); t=9 is the lazy provider. ville_bound_mean = mean over subjects "
                     "of 1 if accused within t canaries else min(1, alpha*W_t): Ville's bound on the eventual accusation "
                     "probability once routing starts.")
    save("audit_route", res)


# ------------------------------------------------------------------------------------------------ 3. post-hoc configuration choice
def config_grid():
    g = [dict(name="default")]
    g += [dict(name=f"M={m}", M=m) for m in (2, 5)]
    g += [dict(name=f"kinds={k}", kinds=[k]) for k in L.KINDS]
    g += [dict(name="kinds=direct+paraphrase", kinds=["direct", "paraphrase"]), dict(name="kinds=d+p+c", kinds=["direct", "paraphrase", "cloze"])]
    g += [dict(name=f"strategy={s}", strategy=s) for s in ["top1_fixed", "mixture_top1", "top1_full", "linear_fixed"]]
    g += [dict(name="method=sum", method="sum"), dict(name="method=rank", method="rank"), dict(name="norm=mean", norm="mean")]
    return g


def ranks_cfg(Vx, a, c):
    """Ranks of subject a (ordering 0) under configuration c (re-uses the analyze.py aggregation conventions)."""
    out = []
    for s in ord0(Vx, a):
        e = Vx.X[(a, s)]
        M = c.get("M", 10)
        if M < 10:
            CAN = L.canaries(Vx.key)
            cols10 = CAN.subset(a, s, 10)
            colsM = CAN.subset(a, s, M)
            pos = [cols10.index(x) for x in colsM]
            p = colsM.index(CAN.planted(a, s))
        else:
            pos, p = list(range(10)), e["p"]
        sel = [i for i, k in enumerate(e["kinds"]) if c.get("kinds") is None or k in c["kinds"]]
        X = e["S"][np.ix_(sel, pos)]
        if c.get("norm") == "mean":
            raise NotImplementedError
        sc = E.aggregate(X, c.get("method", "z"))
        out.append((L.tie_rank(sc, p, stable_int("tie", Vx.vid, a, s, M)), M))
    return out


def part_config_select():
    grid = [c for c in config_grid() if c.get("norm") != "mean"]
    procs = []  # null processes, ordering 0
    for key, name in ALLTRAINED:
        Vx = V(key, name)
        roles = ["phantom"] + (["deleted"] if (key, name) in COMPLIANT else [])
        for role in roles:
            for a in Vx.subjects(role):
                acc = []
                for c in grid:
                    rk = ranks_cfg(Vx, a, c)
                    W = E.eprocess_path([r for r, _ in rk], rk[0][1], c.get("strategy", "plugin"))
                    acc.append(L.first_cross(W, ALPHA) is not None)
                procs.append(dict(key=key, variant=name, role=role, subject=a, acc=acc))
    A = np.array([p["acc"] for p in procs])
    res = dict(grid=[c["name"] for c in grid], n_processes=len(procs),
               default=L.rate(int(A[:, 0].sum()), len(A)),
               any_config=L.rate(int(A.any(1).sum()), len(A)),
               per_config={c["name"]: L.rate(int(A[:, i].sum()), len(A)) for i, c in enumerate(grid)})
    rng = np.random.default_rng(0)
    res["any_of_G"] = {}
    for G in [2, 4, 8, len(grid)]:
        vals = []
        for _ in range(200):
            idx = np.r_[0, rng.choice(np.arange(1, len(grid)), G - 1, replace=False)]
            vals.append(A[:, idx].any(1).mean())
        res["any_of_G"][str(G)] = dict(mean_rate=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))
    comp = [i for i, p in enumerate(procs) if p["role"] == "deleted" and p["variant"].endswith("retrain")]
    res["compliant_deleted_default"] = L.rate(int(A[comp, 0].sum()), len(comp))
    res["compliant_deleted_any"] = L.rate(int(A[comp].any(1).sum()), len(comp))
    res["_notes"] = ("Null processes in ordering 0 (phantoms on all 33 trained/never-trained variants of the 3 keys + deleted subjects "
                     "on the 4 compliant retrains and the never-trained model). any_config = accused by at least one of the "
                     f"{len(grid)} verifier configurations of the paper's ablation grid, i.e. a subject who chooses the configuration "
                     "after seeing the transcript. any_of_G: default plus G-1 random other configurations (200 draws).")
    save("config_select", res)


# ------------------------------------------------------------------------------------------------ 4. key grinding by a malicious subject
def part_grinding():
    Vb = V("key0", "qwen3b__base")
    res = {}
    rng = np.random.default_rng(1)
    for target in ["qwen3b__retrain", "llama3b__retrain"]:
        Vt = V("key0", target)
        subs = [a for a in Vb.subjects("deleted") if a in Vt.subjects("deleted")]

        def rank_table(Vx, a):
            T = []
            for s in ord0(Vx, a):
                sc = L.zagg(Vx.X[(a, s)]["S"])
                T.append([L.tie_rank(sc, p, stable_int("grind", Vx.vid, a, s, p)) for p in range(10)])
            return np.array(T)  # (9, 10): rank of position p
        tabs = {a: (rank_table(Vb, a), rank_table(Vt, a)) for a in subs}
        out = {}
        for G in [1, 10, 100, 1000, 10000]:
            acc, accb = [], []
            for rep in range(20):
                for a in subs:
                    Rb, Rt = tabs[a]
                    P = rng.integers(0, 10, size=(G, Rb.shape[0]))
                    rb = Rb[np.arange(Rb.shape[0])[None, :], P]
                    Wb = E.eprocess_paths_vec(rb, 10, "plugin")
                    best = int(np.argmax(Wb.max(1)))
                    rt = Rt[np.arange(Rt.shape[0]), P[best]]
                    acc.append(L.first_cross(E.eprocess_path(list(rt), 10, "plugin"), ALPHA) is not None)
                    accb.append(bool(Wb[best].max() >= 1 / ALPHA))
            out[str(G)] = dict(false_accusation_target=float(np.mean(acc)), accused_on_public_model=float(np.mean(accb)),
                               n=len(acc), ci95=E.clopper_pearson(int(np.sum(acc)), len(acc)))
        res[target] = out
    res["_notes"] = ("Malicious subject grinds G keys offline against the public never-trained Qwen2.5-3B (the planted positions "
                     "of the 9 canaries within the fixed scored subsets are drawn per key; real grinding also changes the candidate "
                     "sets, so this is a lower bound on the attack), keeps the key with the largest wealth on the public model, and "
                     "then audits a compliant retrain that never saw the documents. 20 repetitions x 20 deleted subjects. "
                     "With pi derived from F_K(.||rho), rho public randomness released after the commitment, G is effectively 1.")
    save("grinding", res)


# ------------------------------------------------------------------------------------------------ 5. baselines (fair twins, fixed-n rank tests, group-sequential)
def rank_sum_null(n, M=10):
    """Exact null pmf of sum of n iid uniform{1..M} ranks."""
    p = np.zeros(n * M + 1); p[0] = 1.0
    for _ in range(n):
        q = np.zeros_like(p)
        for r in range(1, M + 1):
            q[r:] += p[:len(p) - r] / M
        p = q
    return p  # index = sum


def rank_sum_crit(n, alpha=ALPHA, M=10):
    p = rank_sum_null(n, M)
    cdf = np.cumsum(p)
    c = int(np.nonzero(cdf <= alpha)[0].max()) if np.any(cdf <= alpha) else -1  # reject if sum <= c
    size = float(cdf[c]) if c >= 0 else 0.0
    gamma = (alpha - size) / p[c + 1]  # randomise at c+1 for exact size
    return c, size, float(gamma)


def binom_crit(n, alpha=ALPHA, p0=0.1):
    for c in range(n + 2):
        if st.binom.sf(c - 1, n, p0) <= alpha:
            size = float(st.binom.sf(c - 1, n, p0))
            gamma = (alpha - size) / st.binom.pmf(c - 1, n, p0) if c >= 1 else 0.0
            return c, size, float(gamma)
    return n + 1, 0.0, 0.0


def gs_boundaries(n=9, alpha=ALPHA, p0=0.1, spend="pocock"):
    """Exact group-sequential boundaries for the top-1 count with looks after every canary (Lan-DeMets spending)."""
    t = np.arange(1, n + 1) / n
    if spend == "pocock":
        a_t = alpha * np.log(1 + (np.e - 1) * t)
    else:  # O'Brien-Fleming type
        a_t = 2 - 2 * st.norm.cdf(st.norm.ppf(1 - alpha / 2) / np.sqrt(t))
    bounds = []
    dist = np.zeros(n + 1); dist[0] = 1.0  # P(K_j = k, not yet rejected)
    spent = 0.0
    for j in range(1, n + 1):
        nd = np.zeros(n + 1)
        nd[1:] += dist[:-1] * p0; nd += dist * (1 - p0)
        dist = nd
        c = j + 1
        for cc in range(0, j + 2):
            if spent + dist[cc:].sum() <= a_t[j - 1] + 1e-12:
                c = cc; break
        spent += dist[c:].sum(); dist[c:] = 0
        bounds.append(int(c))
    return bounds, float(spent)


def part_baselines2():
    CANs = {k: L.canaries(k) for k in L.CFG}
    tests = {}
    crit9 = rank_sum_crit(9); bc9 = binom_crit(9)
    gsp, gsp_size = gs_boundaries(spend="pocock"); gso, gso_size = gs_boundaries(spend="obf")
    rs_crit = {n: rank_sum_crit(n) for n in range(1, 10)}
    bc = {n: binom_crit(n) for n in range(1, 10)}
    rng = np.random.default_rng(7)
    q = E.geom_prior(10)

    def evaluate(Vx, a):
        sl = ord0(Vx, a)
        e0 = [Vx.X[(a, s)] for s in sl]
        rk = [L.canary_rank(Vx, a, s) for s in sl]
        n = len(rk)
        hits = [int(r == 1) for r in rk]
        out = {}
        W = E.eprocess_path(rk, 10, "plugin")
        for al in (0.05, 0.01, 0.001):
            out[f"eproc@{al}"] = L.first_cross(W, al)
        # fixed prior-shaped bets (anytime-valid, no learning)
        for lam in (0.5, 0.75, 1.0):
            f = (1 - lam) + lam * 10 * q
            out[f"prior_bet_{lam}"] = L.first_cross(np.r_[1.0, np.cumprod([f[r - 1] for r in rk])], ALPHA)
        out["top1_fixed_0.5"] = L.first_cross(E.eprocess_path(rk, 10, "top1_fixed", lam=0.5), ALPHA)
        out["kind_mix"] = L.first_cross(wealth_agg(Vx, a, "kind_mix"), ALPHA)
        out["kind_avg"] = L.first_cross(wealth_agg(Vx, a, "kind_avg"), ALPHA)
        out["binom_peek"] = E.peeking_binom_first(hits, 0.1, ALPHA)
        # score-based predictable bet: f = M softmax(lam * S_j) at the planted candidate (valid by Lemma 1)
        for lam in (1.0, 2.0, "mix"):
            Wv = [1.0]; Wm = np.ones(4); grid = np.array([0.5, 1.0, 2.0, 4.0])
            for ee in e0:
                s = L.zagg(ee["S"])
                if lam == "mix":
                    P = np.exp(grid[:, None] * (s - s.max())[None, :]); P /= P.sum(1, keepdims=True)
                    Wm = Wm * 10 * P[:, ee["p"]]; Wv.append(float(Wm.mean()))
                else:
                    P = np.exp(lam * (s - s.max())); P /= P.sum()
                    Wv.append(Wv[-1] * 10 * P[ee["p"]])
            out[f"score_bet_{lam}"] = L.first_cross(np.array(Wv), ALPHA)
        # fixed-n tests on the 9 ranks
        if n == 9:
            ssum = sum(rk)
            c, size, gam = crit9
            out["ranksum_fixed"] = 9 if ssum <= c else None
            out["ranksum_fixed_rand"] = 9 if (ssum <= c or (ssum == c + 1 and rng.random() < gam)) else None
            k = sum(hits); cb, sb, gb = bc9
            out["binom_fixed"] = 9 if k >= cb else None
            out["binom_fixed_rand"] = 9 if (k >= cb or (k == cb - 1 and rng.random() < gb)) else None
            pmid = st.binom.sf(k, 9, 0.1) + 0.5 * st.binom.pmf(k, 9, 0.1)
            out["binom_midp"] = 9 if pmid <= ALPHA else None
            # curtailed fixed-n tests: same decision as the fixed-n test, stop as soon as it is determined
            K = np.cumsum(hits); Ssum = np.cumsum(rk)
            rej = [j + 1 for j in range(9) if K[j] >= cb]
            acc = [j + 1 for j in range(9) if K[j] + (9 - j - 1) < cb]
            out["binom_curtailed"] = rej[0] if rej else None
            out["binom_curtailed_stop"] = min((rej or [99])[0], (acc or [99])[0], 9)
            rej = [j + 1 for j in range(9) if Ssum[j] + (9 - j - 1) * 10 <= c]   # rejection guaranteed whatever the later ranks
            acc = [j + 1 for j in range(9) if Ssum[j] + (9 - j - 1) * 1 > c]     # acceptance guaranteed
            out["ranksum_curtailed"] = rej[0] if rej else None
            out["ranksum_curtailed_stop"] = min((rej or [99])[0], (acc or [99])[0], 9)
            # group-sequential top-1 binomial (9 looks)
            for nm, bd in (("gs_pocock", gsp), ("gs_obf", gso)):
                rj = [j + 1 for j in range(9) if K[j] >= bd[j]]
                out[nm] = rj[0] if rj else None
        out["n"] = n
        return out

    def twins(Vx, a):
        CAN = CANs[Vx.key]
        res = {}
        x_g, x_raw = [], []
        for s in ord0(Vx, a):
            ee = Vx.X[(a, s)]
            cols10 = CAN.subset(a, s, 10)
            d0 = [c for c in CAN.subset(a, s, 2) if c != CAN.planted(a, s)][0]
            dpos = cols10.index(d0)
            sc = L.zagg(ee["S"])  # M=10-standardised scores (Prop. 3 garbling)
            x_g.append(int(sc[ee["p"]] > sc[dpos] or (sc[ee["p"]] == sc[dpos] and stable_int("tw", a, s) % 2 == 0)))
            diff = (ee["S"][:, ee["p"]] - ee["S"][:, dpos]).sum()
            x_raw.append(int(diff > 0 or (diff == 0 and stable_int("tw", a, s) % 2 == 0)))
        for nm, x in (("garbled", x_g), ("rawdiff", x_raw)):
            res[f"twin_{nm}_ppr"] = L.first_cross(E.twin_ppr_path(x), ALPHA)
            res[f"twin_{nm}_eproc"] = L.first_cross(E.eprocess_path([2 - v for v in x], 2, "plugin"), ALPHA)
            res[f"twin_{nm}_fixedn"] = len(x) if E.fixed_n_binom_reject(x, len(x), 0.5, ALPHA) else None
        return res

    def budget(Vx, a):
        """Equal scoring-call budgets per canary: M=10 x 4 probes (40 calls), M=5 x 8 probes (40), M=2 x 16 probes (32)."""
        CAN = CANs[Vx.key]
        res = {}
        p4 = {"direct0", "paraphrase0", "cloze0", "fewshot0"}
        p8 = p4 | {"direct1", "paraphrase1", "cloze1", "fewshot1"}
        for nm, M, pids in (("M10_k4", 10, p4), ("M5_k8", 5, p8), ("M2_k16", 2, None), ("M10_k16", 10, None), ("M5_k16", 5, None)):
            rk = []
            for s in ord0(Vx, a):
                ee = Vx.X[(a, s)]
                cols10 = CAN.subset(a, s, 10); colsM = CAN.subset(a, s, M)
                pos = [cols10.index(x) for x in colsM]
                sel = [i for i, q_ in enumerate(ee["pids"]) if pids is None or q_ in pids]
                sc = L.zagg(ee["S"][np.ix_(sel, pos)])
                rk.append(L.tie_rank(sc, colsM.index(CAN.planted(a, s)), stable_int("budget", Vx.vid, a, s, nm)))
            res[nm] = L.first_cross(E.eprocess_path(rk, M, "plugin"), ALPHA)
        return res

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    rows = []
    for key, name in ALLTRAINED:
        Vx = V(key, name)
        roles = ["deleted", "phantom"]
        for role in roles:
            for a in Vx.subjects(role):
                o = evaluate(Vx, a)
                if role == "deleted":
                    o.update(twins(Vx, a)); o.update(budget(Vx, a))
                o.update(key=key, variant=name, role=role, subject=a, group=group_of(name),
                         null=(role == "phantom" or (key, name) in COMPLIANT))
                rows.append(o)
    tests_list = sorted({k for r in rows for k in r if k not in ("key", "variant", "role", "subject", "group", "null", "n")
                         and not k.endswith("_stop")})
    res = dict(constants=dict(ranksum_crit9=crit9, binom_crit9=bc9, gs_pocock=(gsp, gsp_size),
                              gs_obf=(gso, gso_size)))
    summ = {}
    for t in tests_list:
        d = {}
        for grp in ["lazy", "npo", "rmu", "graddiff", "adaptive50", "adaptive90", "ga", "compliant", "never_trained", "all_nonnull"]:
            rr = [r for r in rows if r["role"] == "deleted" and t in r and (r["group"] == grp or (grp == "all_nonnull" and not r["null"]))]
            if rr:
                k = sum(r[t] is not None for r in rr)
                d[grp] = L.rate(k, len(rr))
        nn = [r for r in rows if r["null"] and t in r and r["n"] == 9]
        d["null_9canary"] = L.rate(sum(r[t] is not None for r in nn), len(nn)) if nn else None
        nn5 = [r for r in rows if r["null"] and t in r and r["n"] == 5]
        d["null_5canary"] = L.rate(sum(r[t] is not None for r in nn5), len(nn5)) if nn5 else None
        summ[t] = d
    res["detection"] = summ
    sel = lambda r: r["key"] == "key0" and r["variant"].startswith("qwen3b")
    res["selection_vs_confirmation"] = {t: dict(
        key0_qwen=L.rate(sum(r[t] is not None for r in rows if r["role"] == "deleted" and not r["null"] and t in r and sel(r)),
                         sum(1 for r in rows if r["role"] == "deleted" and not r["null"] and t in r and sel(r))),
        heldout=L.rate(sum(r[t] is not None for r in rows if r["role"] == "deleted" and not r["null"] and t in r and not sel(r)),
                       sum(1 for r in rows if r["role"] == "deleted" and not r["null"] and t in r and not sel(r))))
        for t in tests_list}
    # canaries used (stopping time, censored at 9) per behaviour for the e-process and the curtailed fixed-n tests
    used = {}
    for grp in ["lazy", "npo", "rmu", "graddiff", "adaptive50", "adaptive90", "ga", "compliant", "never_trained"]:
        rr = [r for r in rows if r["role"] == "deleted" and r["group"] == grp and r["n"] == 9]
        if not rr:
            continue
        used[grp] = dict(n=len(rr),
                         eproc=float(np.mean([r["eproc@0.05"] or 9 for r in rr])),
                         binom_curtailed=float(np.mean([r["binom_curtailed_stop"] for r in rr])),
                         ranksum_curtailed=float(np.mean([r["ranksum_curtailed_stop"] for r in rr])),
                         gs_pocock=float(np.mean([r["gs_pocock"] or 9 for r in rr])),
                         kind_mix=float(np.mean([r["kind_mix"] or 9 for r in rr])))
        used[grp]["saving_vs_fixed9"] = 1 - used[grp]["eproc"] / 9
    res["canaries_used"] = used
    # cluster sign-flip permutation test (clusters = (key, subject)) for ours vs alternatives on the non-null audits
    def perm(t1, t2, B=20000):
        rr = [r for r in rows if r["role"] == "deleted" and not r["null"] and t1 in r and t2 in r]
        cl = collections.defaultdict(float)
        for r in rr:
            cl[(r["key"], r["subject"])] += (r[t1] is not None) - (r[t2] is not None)
        d = np.array(list(cl.values()))
        obs = d.sum()
        g = np.random.default_rng(3)
        sims = (g.choice([-1, 1], size=(B, len(d))) * d).sum(1)
        return dict(n_clusters=len(d), diff=float(obs), p_two_sided=float((np.abs(sims) >= abs(obs) - 1e-9).mean()))
    res["cluster_perm"] = {f"eproc_vs_{t}": perm("eproc@0.05", t) for t in
                           ["binom_fixed", "ranksum_fixed", "twin_garbled_ppr", "twin_rawdiff_ppr", "twin_garbled_fixedn", "prior_bet_0.5",
                            "score_bet_mix", "kind_mix", "kind_avg", "gs_pocock"]}
    save("baselines2", res)
    save_json(rows, os.path.join(OUT, "baselines2_rows.json"), indent=None)


# ------------------------------------------------------------------------------------------------ 6. subject-level MIA
def part_mia2():
    res = {}
    import glob
    for f in sorted(glob.glob(os.path.join(L.ROOT, "results/main/mia/*.jsonl"))) + sorted(glob.glob(os.path.join(L.ROOT, "results/rev1/mia/*.jsonl"))):
        name = os.path.basename(f)[:-6]
        rows = read_jsonl(f)
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["subject"], r["role"])].append(r["minkpp"])
        ph = {a: np.mean(v) for (a, role), v in by.items() if role == "phantom"}
        out = {}
        for role in ["deleted", "phantom"]:
            ps = []
            for (a, rl), v in by.items():
                if rl != role:
                    continue
                m = np.mean(v)
                pool = [x for b, x in ph.items() if b != a]
                ps.append((1 + sum(x >= m for x in pool)) / (1 + len(pool)))
            k = sum(p <= ALPHA for p in ps)
            out[role] = dict(L.rate(k, len(ps)), min_p=float(min(ps)) if ps else None, mean_p=float(np.mean(ps)) if ps else None)
        # the paper's per-document sequential product, for comparison
        m = mia_baseline(rows, [ALPHA])
        out["doc_product_phantom"] = L.rate(sum(x[f"rej@{ALPHA}"] for x in m if x["role"] == "phantom"), sum(1 for x in m if x["role"] == "phantom"))
        out["doc_product_deleted"] = L.rate(sum(x[f"rej@{ALPHA}"] for x in m if x["role"] == "deleted"), sum(1 for x in m if x["role"] == "deleted"))
        res[name] = out
    res["_notes"] = ("Subject-level conformal MIA: one p-value per subject = rank of the subject's mean Min-K%++ among the other "
                     "phantom subjects' means (leave-one-out, n_ph = 19 or 20; smallest attainable p = 1/20 or 1/21). Exactly valid "
                     "for phantoms (exchangeable at subject level); for real subjects valid only if they are exchangeable with phantoms.")
    save("mia2", res)


# ------------------------------------------------------------------------------------------------ 7. Type-I: clustered CIs, 5- vs 9-canary nulls, 9-canary phantom audits
def part_typeI2():
    procs = []
    for key, name in ALLTRAINED:
        Vx = V(key, name)
        roles = ["phantom"] + (["deleted"] if (key, name) in COMPLIANT else [])
        for role in roles:
            for a in Vx.subjects(role):
                for o in range(3):
                    W = wealth_default(Vx, a, o)
                    procs.append(dict(key=key, variant=name, role=role, subject=a, order=o, n=len(W) - 1,
                                      acc=L.first_cross(W, ALPHA) is not None, acc01=L.first_cross(W, 0.01) is not None))
    def cl_boot(ps, keyf, B=4000, fld="acc"):
        g = collections.defaultdict(list)
        for p in ps:
            g[keyf(p)].append(p[fld])
        G = list(g.values()); kc = np.array([sum(x) for x in G], float); nc = np.array([len(x) for x in G], float)
        idx = np.random.default_rng(0).integers(0, len(G), (B, len(G)))
        r = kc[idx].sum(1) / nc[idx].sum(1)
        return dict(n_clusters=len(G), ci95=[float(np.quantile(r, .025)), float(np.quantile(r, .975))])
    res = {}
    for nm, ps in [("all_orderings", procs), ("ordering0", [p for p in procs if p["order"] == 0])]:
        k = sum(p["acc"] for p in ps)
        res[nm] = dict(L.rate(k, len(ps)),
                       cluster_key_subject=cl_boot(ps, lambda p: (p["key"], p["subject"], p["role"] == "phantom")),
                       n_variants=len({(p["key"], p["variant"]) for p in ps}))
        for ncan in (5, 9):
            pp = [p for p in ps if p["n"] == ncan]
            res[nm][f"{ncan}_canary"] = dict(L.rate(sum(p["acc"] for p in pp), len(pp)),
                                              cluster_key_subject=cl_boot(pp, lambda p: (p["key"], p["subject"], p["role"] == "phantom")))
    mc = load_json(os.path.join(L.ROOT, "results/main/analysis/mc_null.json"))
    res["mc_size"] = {"5_canary": mc["M=10,n=5,plugin,alpha=0.05"]["rate"], "9_canary": mc["M=10,n=9,plugin,alpha=0.05"]["rate"]}
    # 9-canary phantom audits: slots 0-4 from the main logs + slots 5-8 from results/rev1_ph9*
    ph9 = {}
    for key, name, sub in [("key0", "qwen3b__base", "rev1_ph9"), ("key0", "qwen3b__retrain", "rev1_ph9"), ("key1", "qwen3b__retrain", "rev1_ph9_key1"),
                           ("key2", "qwen3b__retrain", "rev1_ph9_key2"), ("key0", "llama3b__retrain", "rev1_ph9"),
                           ("key0", "qwen3b__planted", "rev1_ph9"), ("key0", "qwen3b__npo", "rev1_ph9")]:
        f = os.path.join(L.ROOT, "results", sub, "probes", f"{name}.jsonl")
        if not os.path.exists(f):
            continue
        extra = [r for r in read_jsonl(f) if r["role"] == "phantom"]
        base = [r for r in read_jsonl(os.path.join(L.ROOT, L.RUNDIR[key], "probes", f"{name}.jsonl")) if r["role"] == "phantom"]
        Vx = V(key, name + "@ph9", rows=base + extra)
        Vx.vid = name.replace("__", "/")
        if any(len(Vx.slots(a)) != 9 for a in Vx.subjects("phantom")):
            ph9[f"{key}/{name}"] = "incomplete"; continue
        k = n = 0; k1 = 0; uni = []
        for a in Vx.subjects("phantom"):
            for o in range(3):
                W = wealth_default(Vx, a, o)
                k += L.first_cross(W, ALPHA) is not None; n += 1
            for s in Vx.slots(a):
                uni.append(L.canary_rank(Vx, a, s))
        cnt = np.bincount(uni, minlength=11)[1:]
        ph9[f"{key}/{name}"] = dict(L.rate(k, n), chi2_p=float(st.chisquare(cnt).pvalue), top1=float(np.mean(np.array(uni) == 1)), n_canaries=len(uni))
    tot = [v for v in ph9.values() if isinstance(v, dict)]
    res["phantom_9canary"] = dict(per_variant=ph9, pooled=L.rate(sum(v["k"] for v in tot), sum(v["n"] for v in tot)) if tot else None)
    save("typeI2", res)


# ------------------------------------------------------------------------------------------------ 8. Theorem 2: predicted vs observed, out of sample
def simulate_detection(laws, n_sims=20000, seed=0, alpha=ALPHA):
    """laws: list of per-position rank laws (len 9) -> P(tau <= n) for n = 1..len(laws)."""
    g = np.random.default_rng(seed)
    R = np.stack([g.choice(np.arange(1, 11), size=n_sims, p=p) for p in laws], 1)
    W = E.eprocess_paths_vec(R, 10, "plugin")
    hit = W >= 1 / alpha
    first = np.where(hit.any(1), hit.argmax(1), 10 ** 6)
    return [float((first <= n).mean()) for n in range(1, len(laws) + 1)]


def part_thm2():
    res = {}
    rows = []
    for key, name in NONNULL + COMPLIANT:
        Vx = V(key, name)
        subs = Vx.subjects("deleted")
        R = {a: L.subject_rank_seq(Vx, a, 0)[0] for a in subs}
        if any(len(v) != 9 for v in R.values()):
            continue
        obs_curve = [float(np.mean([L.first_cross(E.eprocess_path(R[a], 10, "plugin")[:n + 1], ALPHA) is not None for a in subs])) for n in range(1, 10)]
        allr = np.array([R[a] for a in subs])
        law = (np.bincount(allr.ravel(), minlength=11)[1:] + 0.5); law = law / law.sum()
        D = E.kl(law, np.full(10, .1))
        iid_curve = simulate_detection([law] * 9)
        # out of sample: estimate on half A (even index), predict half B, and vice versa; per-position (heterogeneous) and iid laws
        pred_h, pred_i, obs_h = [], [], []
        idx = np.arange(len(subs))
        for A_, B_ in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
            RA, RB = allr[A_], allr[B_]
            pos_laws = []
            groups = [range(0, 5), range(5, 7), range(7, 9)]  # ordering 0 = planting-rate order (4,4,4,4,4,2,2,1,1)
            glaw = []
            for gidx in groups:
                c = np.bincount(RA[:, list(gidx)].ravel(), minlength=11)[1:] + 0.1; glaw.append(c / c.sum())
            for j in range(9):
                pos_laws.append(glaw[0] if j < 5 else glaw[1] if j < 7 else glaw[2])
            c = np.bincount(RA.ravel(), minlength=11)[1:] + 0.5
            pred_h.append(simulate_detection(pos_laws, 10000)[-1]); pred_i.append(simulate_detection([c / c.sum()] * 9, 10000)[-1])
            obs_h.append(float(np.mean([L.first_cross(E.eprocess_path(list(r), 10, "plugin"), ALPHA) is not None for r in RB])))
        used = [min(L.first_cross(E.eprocess_path(R[a], 10, "plugin"), ALPHA) or 9, 9) for a in subs]
        p1 = float(np.mean(allr == 1))  # raw top-1 rate (as in Thm 2(iii))
        G1 = E.growth_top1(p1, 10)
        rows.append(dict(key=key, variant=name, group=group_of(name), D=D, p1=p1, accused_all_orderings=float(np.mean(
                             [L.first_cross(wealth_default(Vx, a, o), ALPHA) is not None for a in subs for o in range(3)])),
                         observed_curve=obs_curve, iid_predicted_curve=iid_curve,
                         oos_pred_heterogeneous=float(np.mean(pred_h)), oos_pred_iid=float(np.mean(pred_i)), oos_observed=float(np.mean(obs_h)),
                         mean_canaries_used_censored9=float(np.mean(used)),
                         lower_bound_canaries=(((1 - 0.1) * math.log(1 / ALPHA) - math.log(2)) / D) if D > 0 else None,
                         wald_kl_canaries=math.log(1 / ALPHA) / D if D > 0 else None,
                         top1_bound_thm2iii=(math.log(1 / ALPHA) + math.log(10 * p1)) / G1 if G1 > 0 else None))
    res["per_variant"] = rows
    oo = np.array([[r["oos_observed"], r["oos_pred_heterogeneous"], r["oos_pred_iid"]] for r in rows])
    nonnull = [r for r in rows if r["group"] not in ("compliant", "never_trained")]
    res["oos_summary"] = dict(n=len(rows), mae_heterogeneous=float(np.abs(oo[:, 0] - oo[:, 1]).mean()), mae_iid=float(np.abs(oo[:, 0] - oo[:, 2]).mean()),
                              pearson_heterogeneous=float(np.corrcoef(oo[:, 0], oo[:, 1])[0, 1]), pearson_iid=float(np.corrcoef(oo[:, 0], oo[:, 2])[0, 1]))
    det = [r["observed_curve"][-1] for r in nonnull]
    res["spearman_nonnull"] = dict(n=len(nonnull), **dict(zip(["rho", "p"], map(float, st.spearmanr([r["D"] for r in nonnull], det)))))
    # finite-sample completeness of the plug-in bet (new Thm 2(v)): smallest n with
    # exp(-2 (n*gamma - Reg_n - log(1/alpha))^2 / (n c^2)) <= beta, Q = (1-eta) P + eta U optimised over eta
    q = E.geom_prior(10); beta_r = 2.0 * q
    def regret(n):
        return (10 + 3) / 2 * math.log(n + 1) + 1 / 12 + float(np.log(1 / beta_r).sum())
    comp = {}
    for r in rows:
        if r["variant"] not in ("qwen3b__planted", "llama3b__planted", "qwen3b__npo", "qwen7b__planted") or r["key"] != "key0":
            continue
        Vx = V(r["key"], r["variant"])
        allr = np.array([L.subject_rank_seq(Vx, a, 0)[0] for a in Vx.subjects("deleted")])
        P = np.bincount(allr.ravel(), minlength=11)[1:] + 0.5; P = P / P.sum()
        best = None
        for eta in np.linspace(0.0, 0.9, 19):
            Q = (1 - eta) * P + eta / 10
            gam = float((P * np.log(10 * Q)).sum()); cc = float(np.log(Q.max() / Q.min()))
            for n in range(1, 5001):
                s_ = n * gam - regret(n) - math.log(1 / ALPHA)
                if s_ > 0 and math.exp(-2 * s_ ** 2 / (n * cc ** 2)) <= 0.1:
                    if best is None or n < best[0]:
                        best = (n, float(eta), gam, cc)
                    break
        comp[r["variant"]] = dict(B_canaries=best[0] if best else None, eta=best[1] if best else None, gamma=best[2] if best else None,
                                  c=best[3] if best else None, D=r["D"], observed_mean_canaries_used=r["mean_canaries_used_censored9"],
                 regret_at_9=regret(9), regret_at_B=regret(best[0]) if best else None)
    res["finite_sample_completeness_beta0.1"] = comp
    save("thm2", res)


# ------------------------------------------------------------------------------------------------ 9. paired McNemar: filters/routes and RAG vs compliant
def mcnemar(x, y):
    b = sum(1 for u, v in zip(x, y) if u and not v); c = sum(1 for u, v in zip(x, y) if v and not u)
    return dict(b=b, c=c, p=float(st.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)


def part_filters_paired():
    res = {}
    for key, tag in [("key0", "qwen3b"), ("key0", "llama3b")]:
        Vc = V(key, f"{tag}__retrain")
        comp = {a: L.first_cross(wealth_default(Vc, a), ALPHA) is not None for a in Vc.subjects("deleted")}
        r = {}
        for rule in ["kw_full", "semantic", "kw_tok"]:
            for nm, rows in [(f"refuse_{rule}", derived_rows(key, tag, rule)), (f"route_{rule}", derived_rows(key, tag, rule, route_from=f"{tag}__retrain"))]:
                Vx = V(key, f"{tag}__{'filter' if nm.startswith('refuse') else 'route'}_{rule}", rows=rows)
                acc = {a: L.first_cross(wealth_default(Vx, a), ALPHA) is not None for a in Vx.subjects("deleted")}
                subs = sorted(set(acc) & set(comp))
                r[nm] = dict(k=sum(acc[a] for a in subs), n=len(subs), compliant_k=sum(comp[a] for a in subs),
                             **mcnemar([acc[a] for a in subs], [comp[a] for a in subs]))
        res[f"{key}/{tag}"] = r
    # RAG (5 canaries x 8 probes), tie seeds as in analyze_rag
    import glob
    feats = load_json(os.path.join(L.ROOT, "results/main/filters/rag_filter_feats.json"))
    rr = {}
    for beh in ["compliant", "lazy", "partial"]:
        rows = read_jsonl(os.path.join(L.ROOT, "results/main/rag", f"probes_{beh}.jsonl"))
        rr[beh] = rows
    lazy = rr["lazy"]
    ret = np.array([feats[r["key"]]["sem"] for r in lazy if r["role"] == "retained" and r["key"] in feats])
    tau = float(np.quantile(ret, 0.95))
    rules = {"kw_full": lambda k: feats[k]["kw_full"], "kw_tok": lambda k: feats[k]["kw_tok"], "semantic": lambda k: feats[k]["sem"] >= tau}
    for nm, f in rules.items():
        rr[f"filter_{nm}"] = [dict(r, refused=True) if (r["key"] in feats and f(r["key"])) else r for r in lazy]
    acc = {}
    for beh, rows in rr.items():
        Vx = L.Variant("key0", rows, f"rag/{beh}")
        acc[beh] = {a: L.first_cross(wealth_default(Vx, a), ALPHA) is not None for a in Vx.subjects("deleted")}
    rag = {}
    for beh in acc:
        if beh == "compliant":
            continue
        subs = sorted(set(acc[beh]) & set(acc["compliant"]))
        rag[beh] = dict(k=sum(acc[beh][a] for a in subs), n=len(subs), compliant_k=sum(acc["compliant"][a] for a in subs),
                        **mcnemar([acc[beh][a] for a in subs], [acc["compliant"][a] for a in subs]))
    res["rag"] = rag
    res["_notes"] = "Exact McNemar (two-sided binomial on discordant pairs), ordering 0, same 20 deleted subjects; replaces the unpaired Fisher tests."
    save("filters_paired", res)


# ------------------------------------------------------------------------------------------------ 10. unseen-stem probes (memorisation realism)
def part_unseen():
    res = {}
    for key, name in NONNULL + COMPLIANT + [("key0", "qwen3b__scrub-same"), ("key0", "qwen3b__planted-1ep"), ("key0", "qwen3b__diluted-1ep")]:
        f = os.path.join(L.ROOT, "results/rev1/probes", f"{name}.jsonl")
        if os.path.exists(f):
            Vx = L.Variant(key, read_jsonl(f), name.replace("__", "/"))
        elif any(x in name for x in ("scrub", "1ep")):
            continue
        else:
            Vx = V(key, name)
        if not Vx.subjects("deleted"):
            continue
        k, n, _ = L.detect(Vx)
        ku, nu, _ = L.detect(Vx, pids=L.UNSEEN)
        kn, nn, _ = L.detect(Vx, pids=L.NAME_FREE)
        top = float(np.mean([L.canary_rank(Vx, a, s) == 1 for a in Vx.subjects("deleted") for s in Vx.slots(a)]))
        topu = float(np.mean([L.canary_rank(Vx, a, s, L.UNSEEN) == 1 for a in Vx.subjects("deleted") for s in Vx.slots(a)]))
        res[f"{key}/{name}"] = dict(all_probes=L.rate(k, n), unseen_probes=L.rate(ku, nu), namefree_probes=L.rate(kn, nn),
                                    top1_all=top, top1_unseen=topu)
    save("unseen", res)


# ------------------------------------------------------------------------------------------------ 11. configuration chosen on key 0, confirmed on keys 1-2
def part_heldout():
    cfgs = {"default": dict(), "cloze_only": dict(pids={"cloze0", "cloze1", "cloze2", "cloze3"}), "top1_fixed_0.5": dict(strategy="top1_fixed"),
            "M5": dict(M=5)}
    res = {}
    for key in ["key0", "key1", "key2"]:
        for v in ["planted", "npo", "ga", "retrain"]:
            Vx = V(key, f"qwen3b__{v}")
            r = {}
            for nm, c in cfgs.items():
                if nm == "M5":
                    k = n = 0
                    for a in Vx.subjects("deleted"):
                        rk = ranks_cfg(Vx, a, dict(M=5))
                        k += L.first_cross(E.eprocess_path([x for x, _ in rk], 5, "plugin"), ALPHA) is not None; n += 1
                else:
                    k, n, _ = L.detect(Vx, pids=c.get("pids"), strategy=c.get("strategy", "plugin"))
                r[nm] = L.rate(k, n)
            res[f"{key}/{v}"] = r
    save("heldout", res)


# ------------------------------------------------------------------------------------------------ 12. sampled answers with whole-word matching
def part_sampled2():
    import re
    CAN = L.canaries("key0")
    res = {}
    for name in ["qwen3b__planted", "qwen3b__retrain", "qwen3b__npo"]:
        rows = read_jsonl(os.path.join(L.ROOT, "results/main/samples", f"{name}.jsonl"))
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["T"], r["subject"], r["slot"])].append(r)
        out = collections.defaultdict(lambda: collections.defaultdict(list))
        for (T, a, sl), rs in by.items():
            cands = CAN.can[str(a)]["slots"][sl]["candidates"]
            cols = CAN.subset(a, sl, 10)
            def norm(x):
                return " " + re.sub(r"[^0-9a-z]+", " ", x.lower()).strip() + " "
            cnt_sub = np.array([sum(cands[i].lower() in r["answer"].lower() for r in rs) for i in cols], float)
            cnt_w = np.array([sum(norm(cands[i]) in norm(r["answer"]) for r in rs) for i in cols], float)
            p = cols.index(CAN.planted(a, sl))
            for nm, cnt in (("substring", cnt_sub), ("whole_word", cnt_w)):
                rk = L.tie_rank(cnt, p, stable_int("samp", T, a, sl))
                out[str(T)][f"{rs[0]['role']}_{nm}_top1"].append(rk == 1)
                out[str(T)][f"{rs[0]['role']}_{nm}_freq"].append(cnt[p] / len(rs))
        res[name] = {T: {k: float(np.mean(v)) for k, v in d.items()} | {"n_canaries": len(d["deleted_whole_word_top1"])} for T, d in out.items()}
    save("sampled2", res)


# ------------------------------------------------------------------------------------------------ 13. detection at small alpha (main-table groups)
def part_small_alpha():
    res = {}
    for key, name in ALLTRAINED:
        Vx = V(key, name)
        d = {}
        for al in (0.05, 0.01, 0.001):
            k, n, taus = L.detect(Vx, alpha=al)
            d[str(al)] = L.rate(k, n)
        res[f"{key}/{name}"] = d
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for (key, name) in ALLTRAINED:
        g = group_of(name) if name.startswith("qwen3b") else name.split("__")[0] + "_" + group_of(name)
        for al, v in res[f"{key}/{name}"].items():
            agg[g][al][0] += v["k"]; agg[g][al][1] += v["n"]
    res["by_group"] = {g: {al: L.rate(*v) for al, v in d.items()} for g, d in agg.items()}
    save("small_alpha", res)


# ------------------------------------------------------------------------------------------------ 14. new GPU variants (scrubbing provider, dilution, GA trajectory, strong RMU)
def part_newvariants():
    import glob
    res = {}
    util = {}
    for f in sorted(glob.glob(os.path.join(L.ROOT, "results/rev1/utility/*.jsonl"))):
        from fmp.analyze import utility_summary
        util[os.path.basename(f)[:-6]] = utility_summary(read_jsonl(f), 200)
    for f in sorted(glob.glob(os.path.join(L.ROOT, "results/rev1/probes/*.jsonl"))):
        name = os.path.basename(f)[:-6]
        Vx = L.Variant("key0", read_jsonl(f), name.replace("__", "/"))
        d = dict(utility=util.get(name))
        for role in ["deleted", "retained", "phantom"]:
            subs = Vx.subjects(role)
            if not subs:
                continue
            k, n, taus = L.detect(Vx, role)
            allo = [L.first_cross(wealth_default(Vx, a, o), ALPHA) is not None for a in subs for o in range(3)]
            km, nm_ = count(Vx, role, "kind_mix")
            ranks = [L.canary_rank(Vx, a, s) for a in subs for s in Vx.slots(a)]
            cnt = np.bincount(ranks, minlength=11)[1:] + 0.5
            d[role] = dict(order0=L.rate(k, n), all_orderings=L.rate(int(sum(allo)), len(allo)), kind_mix_order0=L.rate(km, nm_),
                           top1=float(np.mean(np.array(ranks) == 1)), D=E.kl(cnt / cnt.sum(), np.full(10, .1)),
                           canaries_used_mean=float(np.mean([t or 9 for t in taus])) if role == "deleted" else None)
            if role == "deleted":
                ku, nu, _ = L.detect(Vx, pids=L.UNSEEN)
                d[role]["unseen_probes_order0"] = L.rate(ku, nu)
                d[role]["by_rate_top1"] = {str(rr): float(np.mean([L.canary_rank(Vx, a, s) == 1 for a in subs for s in Vx.slots(a)
                                                                  if Vx.X[(a, s)]["rate"] == rr])) for rr in (1, 2, 4)}
        res[name] = d
    # paired comparison with the compliant retrain and the lazy model on the same 20 subjects
    Vc, Vl = V("key0", "qwen3b__retrain"), V("key0", "qwen3b__planted")
    comp = {a: L.first_cross(wealth_default(Vc, a), ALPHA) is not None for a in Vc.subjects("deleted")}
    lazy = {a: L.first_cross(wealth_default(Vl, a), ALPHA) is not None for a in Vl.subjects("deleted")}
    for name in res:
        f = os.path.join(L.ROOT, "results/rev1/probes", f"{name}.jsonl")
        Vx = L.Variant("key0", read_jsonl(f), name.replace("__", "/"))
        acc = {a: L.first_cross(wealth_default(Vx, a), ALPHA) is not None for a in Vx.subjects("deleted")}
        subs = sorted(acc)
        res[name]["vs_compliant"] = mcnemar([acc[a] for a in subs], [comp[a] for a in subs])
        res[name]["vs_lazy"] = mcnemar([acc[a] for a in subs], [lazy[a] for a in subs])
    from fmp.analyze import utility_summary
    for v in ["planted", "retrain", "ga", "rmu", "npo"]:
        res.setdefault("_reference_utility", {})[v] = utility_summary(read_jsonl(os.path.join(L.ROOT, "results/main/utility", f"qwen3b__{v}.jsonl")), 200)
    st_ = os.path.join(L.ROOT, "data/derived/full_key0/extra_corpora_stats.json")
    res["_corpora"] = load_json(st_) if os.path.exists(st_) else None
    save("newvariants", res)


def part_main_extra():
    """Per-replicate extra columns for Table I: ordering-0 counts with the kind-mixture aggregator and at alpha = 0.01, and the
    mean number of canaries used (stopping time censored at 9, all subjects)."""
    res = {}
    run = {"key0": "main", "key1": "key1", "key2": "key2"}
    for key, name in ALLTRAINED:
        Vx = V(key, name)
        k, n, taus = L.detect(Vx)
        k01, _, _ = L.detect(Vx, alpha=0.01)
        km, _ = count(Vx, "deleted", "kind_mix")
        res[f"{run[key]}:{name.replace('__', '/')}"] = dict(k=k, n=n, k_alpha01=k01, k_kindmix=km,
                                                            canaries_used_mean=float(np.mean([t or 9 for t in taus])))
    save("main_extra", res)


PARTS = ["routing", "audit_route", "config_select", "grinding", "baselines2", "mia2", "typeI2", "thm2", "filters_paired", "unseen",
         "heldout", "sampled2", "small_alpha", "newvariants", "main_extra"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in a.parts or PARTS:
        if not a.force and os.path.exists(os.path.join(OUT, f"{p}.json")):
            print(f"[revise1] skip {p} (exists)"); continue
        t = time.time()
        globals()[f"part_{p}"]()
        print(f"[revise1] {p} took {time.time() - t:.0f}s", flush=True)
