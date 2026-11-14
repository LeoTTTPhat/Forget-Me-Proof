"""Offline analysis of all probe logs (CPU only): ranks, e-processes, Type-I/power/queries-to-decision,
baselines (twins, binomial, peeking, MIA, traps), filters, sampled answers, RAG, ablations, Monte Carlo null.

python src/fmp/analyze.py --config CFG            -> results/<run>/analysis/*.json
Variant files: results/<run>/probes/<tag>__<variant>.jsonl (tag = model/key/seed family, e.g. qwen3b).
"""
import argparse, collections, glob, itertools, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import run_dir, load_config, read_jsonl, load_json, save_json, stable_int
from fmp.corpus import derived_dir, master_key
from fmp import canary as C, eprocess as E
import numpy as np

# behaviour class of each variant suffix (suffixes may carry -s<seed>)
BEHAVIOUR = {"base": "never_trained", "retrain": "compliant", "planted": "lazy", "ga": "approx", "graddiff": "approx",
             "npo": "approx", "rmu": "approx", "adaptive50": "adaptive", "adaptive90": "adaptive"}
NULL_BEHAVIOURS = {"never_trained", "compliant"}
KINDS = ["direct", "paraphrase", "cloze", "fewshot"]


def behaviour(variant):
    v = variant.split("-s")[0]
    if v.startswith("filter_") or v.startswith("route_"):
        return "suppressive"
    return BEHAVIOUR.get(v, "other")


class Canaries:
    def __init__(self, cfg):
        dd = derived_dir(cfg)
        self.cfg = cfg
        self.subj = {s["id"]: s for s in load_json(os.path.join(dd, "subjects.json"))}
        self.can = load_json(os.path.join(dd, "SECRET_canaries.json"))
        self.master = master_key(cfg)

    def planted(self, a, slot):
        return self.can[str(a)]["slots"][slot]["planted_idx"]

    def subset(self, a, slot, M):
        s = self.can[str(a)]["slots"][slot]
        return C.subsample_with_planted(C.subject_key(self.master, a), a, slot, len(s["candidates"]), s["planted_idx"], M)


def load_table(path, refused_keys=None, route=None):
    """-> {(subject, slot): dict(role, type, rate, cand_idx, pids, kinds, S (P x n) sum-lp, N (P x n) n_tok)}"""
    rows = read_jsonl(path) if isinstance(path, str) else path
    route = route or {}
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("slot") == "trap":
            continue
        if refused_keys is not None and r["key"] in refused_keys:
            if r["key"] in route:
                r = dict(route[r["key"]])
            else:
                r = dict(r, refused=True)
        groups[(r["subject"], r["slot"])].append(r)
    T = {}
    for k, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["pid"])
        S = np.array([[c[0] for c in r["scores"]] for r in rs], float)
        N = np.array([[c[1] for c in r["scores"]] for r in rs], float)
        ref = np.array([bool(r.get("refused")) for r in rs])
        S[ref] = 0.0
        T[k] = dict(role=rs[0]["role"], type=rs[0].get("type"), rate=rs[0].get("rate"), cand_idx=list(rs[0]["cand_idx"]),
                    pids=[r["pid"] for r in rs], kinds=[r["kind"] for r in rs], S=S, N=N, refused=ref)
    return T


def canary_scores(entry, cols, method="z", kinds=None, ref_entry=None, norm="sum", weights_by_kind=None):
    """Aggregate probes of one canary into a score per candidate (restricted to columns `cols` = cand indices)."""
    pos = [entry["cand_idx"].index(c) for c in cols]
    sel = [i for i, k in enumerate(entry["kinds"]) if kinds is None or k in kinds]
    if not sel:
        return np.zeros(len(cols))
    X = entry["S"][np.ix_(sel, pos)]
    if norm == "mean":
        X = X / np.maximum(entry["N"][np.ix_(sel, pos)], 1)
    ref = None
    if ref_entry is not None:
        rpos = [ref_entry["cand_idx"].index(c) for c in cols]
        rmap = {p: i for i, p in enumerate(ref_entry["pids"])}
        ridx = [rmap[entry["pids"][i]] for i in sel]
        ref = ref_entry["S"][np.ix_(ridx, rpos)]
        if norm == "mean":
            ref = ref / np.maximum(ref_entry["N"][np.ix_(ridx, rpos)], 1)
        ref = np.where(entry["refused"][sel][:, None], 0.0, ref)
    w = None
    if weights_by_kind is not None:
        w = np.array([weights_by_kind.get(entry["kinds"][i], 1.0) for i in sel])
    return E.aggregate(X, method, ref=ref, weights=w)


def rank_of(scores, cols, planted, seed):
    rng = np.random.default_rng(seed)
    return E.planted_rank(scores, cols.index(planted), rng)


def subject_canaries(T, a):
    return sorted(sl for (s, sl) in T if s == a)


def ordering(slots, a, o):
    if o == 0:
        return list(slots)
    r = np.random.default_rng(stable_int("order", a, o) % (2 ** 32))
    return [slots[i] for i in r.permutation(len(slots))]


DEFAULT = dict(M=10, method="z", kinds=None, ref=False, norm="sum", strategy="plugin", n_canaries=None, adaptive_kinds=False)


def compute_ranks(T, CAN, cfgv, ref_T=None, tag=""):
    """ranks[(a, slot)] for one verifier configuration (non-adaptive aggregation)."""
    out = {}
    for (a, sl), e in T.items():
        M = min(cfgv["M"], len(e["cand_idx"]))
        cols = CAN.subset(a, sl, M) if M < len(CAN.can[str(a)]["slots"][sl]["candidates"]) else list(range(len(CAN.can[str(a)]["slots"][sl]["candidates"])))
        if not set(cols) <= set(e["cand_idx"]):
            continue
        re_ = ref_T.get((a, sl)) if (cfgv["ref"] and ref_T) else None
        if cfgv["ref"] and re_ is None:
            continue
        s = canary_scores(e, cols, cfgv["method"], cfgv["kinds"], re_, cfgv["norm"])
        out[(a, sl)] = (rank_of(s, cols, CAN.planted(a, sl), stable_int("tie", tag, a, sl, cfgv["M"]) % (2 ** 32)), M)
    return out


def adaptive_kind_ranks(T, CAN, slots_order, a, M, eta=1.0, tag=""):
    """Predictable probe-type weights: w_t ∝ exp(eta * mean planted z-score of kind t on earlier canaries)."""
    hist = collections.defaultdict(list)
    ranks = []
    for sl in slots_order:
        e = T[(a, sl)]
        cols = CAN.subset(a, sl, M)
        w = {k: math.exp(eta * float(np.mean(hist[k]))) if hist[k] else 1.0 for k in KINDS}
        s = canary_scores(e, cols, "z", weights_by_kind=w)
        p = cols.index(CAN.planted(a, sl))
        ranks.append(E.planted_rank(s, p, np.random.default_rng(stable_int("tie", tag, a, sl, M, "ak") % (2 ** 32))))
        for k in KINDS:
            sk = canary_scores(e, cols, "z", kinds=[k])
            if np.any(sk != 0):
                hist[k].append(float(sk[p]))
    return ranks


def subject_processes(T, CAN, cfgv, ranks, alphas, n_order, tag=""):
    """One e-process per (subject, ordering). Returns list of dicts."""
    res = []
    subjects = sorted({a for (a, _) in T})
    for a in subjects:
        slots = [sl for sl in subject_canaries(T, a) if (a, sl) in ranks] if not cfgv["adaptive_kinds"] else subject_canaries(T, a)
        if not slots:
            continue
        if cfgv["n_canaries"]:
            slots = slots[:cfgv["n_canaries"]]
        role = T[(a, slots[0])]["role"]
        nprobe = int(np.mean([len([k for k in T[(a, sl)]["kinds"] if cfgv["kinds"] is None or k in cfgv["kinds"]]) for sl in slots]))
        for o in range(n_order):
            sl_o = ordering(slots, a, o)
            if cfgv["adaptive_kinds"]:
                rk = adaptive_kind_ranks(T, CAN, sl_o, a, cfgv["M"], tag=tag)
                M = cfgv["M"]
            else:
                rk = [ranks[(a, sl)][0] for sl in sl_o]
                M = ranks[(a, sl_o[0])][1]
            W = E.eprocess_path(rk, M, cfgv["strategy"])
            rec = dict(subject=a, role=role, order=o, n=len(rk), M=M, probes_per_canary=nprobe, logW=float(np.log(max(W[-1], 1e-300))),
                       top1=float(np.mean(np.array(rk) == 1)))
            for al in alphas:
                fc = E.first_crossing(W, al)
                rec[f"rej@{al}"] = fc is not None
                rec[f"tau@{al}"] = fc
            res.append(rec)
    return res


def cluster_bootstrap(ps, al, B=2000, seed=0, with_median=False, key=lambda p: p["subject"]):
    """Percentile bootstrap over clusters (default: subjects) of the rejection rate (vectorised) and, optionally, of the
    median canaries-to-decision among rejected processes. Deterministic (fixed seed)."""
    groups = collections.defaultdict(list)
    for p in ps:
        groups[key(p)].append(p)
    G = list(groups.values())
    if len(G) < 2:
        return None
    rng = np.random.default_rng(seed)
    kc = np.array([sum(p[f"rej@{al}"] for p in g) for g in G], float)
    nc = np.array([len(g) for g in G], float)
    idx = rng.integers(0, len(G), (B, len(G)))
    rates = kc[idx].sum(1) / nc[idx].sum(1)
    out = dict(n_clusters=len(G), B=B, reject_rate_ci95=[float(np.quantile(rates, .025)), float(np.quantile(rates, .975))])
    if with_median:
        tau = [[p[f"tau@{al}"] for p in g if p[f"tau@{al}"] is not None] for g in G]
        meds = []
        for row in idx:
            tt = [t for i in row for t in tau[i]]
            if tt:
                meds.append(float(np.median(tt)))
        out["canaries_to_decision_median_ci95"] = [float(np.quantile(meds, .025)), float(np.quantile(meds, .975))] if len(meds) >= B // 2 else None
    return out


def summarize_processes(procs, alphas, role, boot_median=False):
    ps = [p for p in procs if p["role"] == role]
    out = dict(n_processes=len(ps), n_subjects=len({p["subject"] for p in ps}))
    if not ps:
        return out
    for al in alphas:
        k = sum(p[f"rej@{al}"] for p in ps)
        taus = [p[f"tau@{al}"] for p in ps if p[f"tau@{al}"] is not None]
        by_order = collections.defaultdict(list)
        for p in ps:
            by_order[p["order"]].append(p[f"rej@{al}"])
        rates = [np.mean(v) for v in by_order.values()]
        out[f"alpha={al}"] = dict(reject_rate=k / len(ps), k=k, ci95=E.clopper_pearson(k, len(ps)),
                                  reject_rate_order_mean=float(np.mean(rates)), reject_rate_order_std=float(np.std(rates)),
                                  canaries_to_decision_median=float(np.median(taus)) if taus else None,
                                  canaries_to_decision_iqr=[float(np.quantile(taus, .25)), float(np.quantile(taus, .75))] if taus else None,
                                  probes_to_decision_median=float(np.median([t * ps[0]["probes_per_canary"] for t in taus])) if taus else None,
                                  scoring_calls_to_decision_median=float(np.median([t * ps[0]["probes_per_canary"] * ps[0]["M"] for t in taus])) if taus else None)
        # Stage 4: the 3 orderings of one subject reuse the same canary ranks, so the n_processes are not independent.
        # Independent-unit views: (i) the pre-registered ordering 0 only (one process per subject, Clopper-Pearson);
        # (ii) a subject-cluster bootstrap CI of the pooled rate and of the median canaries-to-decision.
        o0 = [p for p in ps if p["order"] == 0]
        k0 = int(sum(p[f"rej@{al}"] for p in o0))
        out[f"alpha={al}"]["order0"] = dict(k=k0, n=len(o0), reject_rate=k0 / len(o0) if o0 else None,
                                            ci95=E.clopper_pearson(k0, len(o0)))
        out[f"alpha={al}"]["subject_cluster_boot"] = cluster_bootstrap(ps, al, with_median=boot_median)
    out["mean_top1"] = float(np.mean([p["top1"] for p in ps]))
    out["mean_logW"] = float(np.mean([p["logW"] for p in ps]))
    return out


def rank_law(ranks, keys, M):
    r = [ranks[k][0] for k in keys if k in ranks]
    if not r:
        return None
    cnt = np.bincount(r, minlength=M + 1)[1:M + 1].astype(float)
    p = (cnt + 0.5) / (cnt.sum() + 0.5 * M)
    pt = float(np.mean(np.array(r) == 1))
    from scipy.stats import chisquare
    return dict(n=len(r), top1=pt, mean_rank=float(np.mean(r)), kl_to_uniform=E.kl(p, np.full(M, 1 / M)),
                chi2_uniform_p=float(chisquare(cnt).pvalue) if len(r) >= 5 * M else None,
                thm2_expected_canaries_top1=E.expected_canaries_bound(pt, M, 0.05) if pt > 1 / M else None,
                hist=cnt.tolist())


# ---------------------------------------------------------------------------------------------- baselines
def twin_and_binomial(T, CAN, alphas, n_order, tag):
    """Huang-style twins (m=1) with PPR martingale and fixed-n binomial; naive peeking binomial on top-1 hits (M=10)."""
    r2 = compute_ranks(T, CAN, dict(DEFAULT, M=2), tag=tag)
    r10 = compute_ranks(T, CAN, dict(DEFAULT, M=10), tag=tag)
    res = []
    for a in sorted({a for (a, _) in T}):
        slots = subject_canaries(T, a)
        role = T[(a, slots[0])]["role"]
        for o in range(n_order):
            so = ordering(slots, a, o)
            x2 = [int(r2[(a, s)][0] == 1) for s in so if (a, s) in r2]
            h10 = [int(r10[(a, s)][0] == 1) for s in so if (a, s) in r10]
            rec = dict(subject=a, role=role, order=o)
            for al in alphas:
                rec[f"twin_ppr@{al}"] = E.first_crossing(E.twin_ppr_path(x2), al) is not None
                rec[f"twin_fixedn@{al}"] = E.fixed_n_binom_reject(x2, len(x2), 0.5, al)
                rec[f"twin_eproc@{al}"] = E.first_crossing(E.eprocess_path([2 - x for x in x2], 2, "plugin"), al) is not None
                rec[f"binom10_fixedn@{al}"] = E.fixed_n_binom_reject(h10, len(h10), 0.1, al)
                rec[f"binom10_peek@{al}"] = E.peeking_binom_first(h10, 0.1, al) is not None
            res.append(rec)
    return res


def mia_baseline(mia_rows, alphas, kappa=0.5):
    """Sequential Min-K%++ MIA: conformal p-values of a subject's documents against phantom (never-trained) documents,
    p-to-e calibrator e = kappa p^(kappa-1), multiplied over documents. No exact null: validity relies on subject and
    phantom documents being exchangeable, which the data subject cannot guarantee."""
    by_s = collections.defaultdict(list)
    for r in mia_rows:
        by_s[(r["subject"], r["role"])].append(r)
    ph = {a: np.array([r["minkpp"] for r in rs]) for (a, role), rs in by_s.items() if role == "phantom"}
    out = []
    for (a, role), rs in sorted(by_s.items()):
        pool = np.concatenate([v for b, v in ph.items() if b != a]) if ph else np.array([])
        if len(pool) == 0:
            continue
        rs = sorted(rs, key=lambda r: r["key"])
        logW, best = 0.0, 0.0
        rej = {al: False for al in alphas}
        for r in rs:
            p = (1 + np.sum(pool >= r["minkpp"])) / (1 + len(pool))
            logW += math.log(kappa) + (kappa - 1) * math.log(p)
            for al in alphas:
                rej[al] = rej[al] or logW >= math.log(1 / al)
        out.append(dict(subject=a, role=role, n_docs=len(rs), logW=logW, mean_minkpp=float(np.mean([r["minkpp"] for r in rs])),
                        **{f"rej@{al}": rej[al] for al in alphas}))
    return out


def trap_baseline(trap_rows, CAN, alphas):
    by = collections.defaultdict(list)
    for r in trap_rows:
        by[r["subject"]].append(r)
    out = []
    for a, rs in by.items():
        P = np.array([[c[0] for c in r["scores"]] for r in rs])
        s = E.aggregate(P, "z")
        pi = CAN.can[str(a)]["trap"]["planted_idx"]
        rk = E.planted_rank(s, pi, np.random.default_rng(stable_int("trap", a) % 2 ** 32))
        M = P.shape[1]
        e = E.PluginBet(M).e(rk)
        out.append(dict(subject=a, role=rs[0]["role"], rank=rk, e=e, **{f"rej@{al}": e >= 1 / al for al in alphas}))
    return out


def rate_summary(rows, keyname, role):
    rs = [r for r in rows if r["role"] == role]
    if not rs:
        return None
    k = sum(bool(r[keyname]) for r in rs)
    return dict(rate=k / len(rs), k=k, n=len(rs), ci95=E.clopper_pearson(k, len(rs)))


# ---------------------------------------------------------------------------------------------- sampled answers
def sampled_analysis(rows, CAN, alphas):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["T"], r["subject"], r["slot"])].append(r)
    res = collections.defaultdict(list)
    for (T, a, sl), rs in by.items():
        cands = CAN.can[str(a)]["slots"][sl]["candidates"]
        cols = CAN.subset(a, sl, 10)
        cnt = np.array([sum(c.lower() in r["answer"].lower() for r in rs) for c in (cands[i] for i in cols)], float)
        rk = rank_of(cnt, cols, CAN.planted(a, sl), stable_int("samp", T, a, sl) % 2 ** 32)
        res[T].append(dict(subject=a, slot=sl, role=rs[0]["role"], rank=rk, any_match=float(np.mean([any(c.lower() in r["answer"].lower() for c in cands) for r in rs])),
                           planted_freq=float(cnt[cols.index(CAN.planted(a, sl))] / len(rs))))
    out = {}
    for T, v in res.items():
        out[str(T)] = {}
        for role in ["deleted", "phantom"]:
            vv = [x for x in v if x["role"] == role]
            if vv:
                out[str(T)][role] = dict(n_canaries=len(vv), top1=float(np.mean([x["rank"] == 1 for x in vv])),
                                         any_candidate_match=float(np.mean([x["any_match"] for x in vv])),
                                         planted_freq=float(np.mean([x["planted_freq"] for x in vv])),
                                         plugin_logW=float(np.log(E.eprocess_path([x["rank"] for x in vv], 10, "plugin")[-1])))
    return out


# ---------------------------------------------------------------------------------------------- utility
def utility_summary(rows, tofu_n=None):
    """Accuracy per split; TOFU splits restricted to the first tofu_n items (config) so variants evaluated before a
    config change (E1 used 400) stay comparable."""
    if tofu_n:
        rows = [r for r in rows if not r["split"].startswith("tofu") or int(r["key"].split("|")[1]) < tofu_n]
    out = {}
    for split in ["tofu_retain", "tofu_forget", "mmlu"]:
        v = [r["correct"] for r in rows if r["split"] == split]
        if v:
            k = int(sum(v)); out[split] = dict(acc=k / len(v), n=len(v), ci95=E.clopper_pearson(k, len(v)))
    return out


# ---------------------------------------------------------------------------------------------- Monte Carlo null (E15)
def mc_null(n_runs, alphas, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for M, n in [(10, 9), (10, 5), (2, 9), (5, 9), (20, 9)]:
        R = rng.integers(1, M + 1, size=(n_runs, n))
        for strat in E.STRATEGIES:
            sup = E.eprocess_paths_vec(R, M, strat).max(1)
            for al in alphas:
                k = int((sup >= 1 / al).sum())
                out[f"M={M},n={n},{strat},alpha={al}"] = dict(rate=k / n_runs, k=k, n=n_runs, ci95=E.clopper_pearson(k, n_runs))
        H = (R == 1).astype(int)
        for al in alphas:
            k = sum(E.peeking_binom_first(h, 1 / M, al) is not None for h in H[: min(n_runs, 20000)])
            nn = min(n_runs, 20000)
            out[f"M={M},n={n},binom_peek,alpha={al}"] = dict(rate=k / nn, k=int(k), n=nn, ci95=E.clopper_pearson(int(k), nn))
    # long-horizon peeking (n = 100 twins) shows the inflation clearly
    X = rng.integers(0, 2, size=(20000, 100))
    for al in alphas:
        k = sum(E.peeking_binom_first(x, 0.5, al) is not None for x in X)
        out[f"twins_n=100,binom_peek,alpha={al}"] = dict(rate=k / 20000, k=int(k), n=20000, ci95=E.clopper_pearson(int(k), 20000))
        k = sum(E.first_crossing(E.twin_ppr_path(x), al) is not None for x in X)
        out[f"twins_n=100,ppr,alpha={al}"] = dict(rate=k / 20000, k=int(k), n=20000, ci95=E.clopper_pearson(int(k), 20000))
    return out


# ---------------------------------------------------------------------------------------------- main
def analyze(cfg, only_mc=False):
    rd = run_dir(cfg)
    out = os.path.join(rd, "analysis")
    os.makedirs(out, exist_ok=True)
    alphas = cfg["stats"]["alpha"]
    n_order = cfg["stats"]["orderings"]
    CAN = Canaries(cfg)
    t0 = time.time()
    mc_path = os.path.join(out, "mc_null.json")
    if not os.path.exists(mc_path):
        save_json(mc_null(cfg["stats"]["mc_runs"], alphas), mc_path)
    if only_mc:
        return
    files = sorted(glob.glob(os.path.join(rd, "probes", "*.jsonl")))
    tags = collections.defaultdict(dict)
    for f in files:
        stem = os.path.basename(f)[:-6]
        tag, var = stem.split("__", 1)
        tags[tag][var] = f
    feats = load_json(os.path.join(rd, "filters", "probe_filter_feats.json")) if os.path.exists(os.path.join(rd, "filters", "probe_filter_feats.json")) else None
    main, ablations, baselines, mem, extras = {}, {}, {}, {}, {}
    all_null_procs = []
    # Filter features are query-text features keyed by probe key and are model-independent (the probe set is shared
    # by all models). Variants whose planted log has no retained-subject probes (e.g. qwen7b, roles deleted+phantom)
    # calibrate the semantic threshold on the retained-subject probe keys of any planted log that has them.
    global_ret_keys = set()
    if feats:
        for _tag, _vars in tags.items():
            if "planted" in _vars:
                global_ret_keys |= {r["key"] for r in read_jsonl(_vars["planted"]) if r["role"] == "retained" and r["key"] in feats}
    global_ret_keys = sorted(global_ret_keys)
    for tag, variants in tags.items():
        tables = {v: load_table(f) for v, f in variants.items()}
        ref_T = tables.get("base")
        # derived suppressive variants from the planted (lazy) logs
        if feats and "planted" in variants:
            raw = read_jsonl(variants["planted"])
            ret_raw = {r["key"]: r for r in read_jsonl(variants["retrain"])} if "retrain" in variants else {}
            ret_keys = [r["key"] for r in raw if r["role"] == "retained" and r["key"] in feats]
            tau_source = "own_retained_probes"
            if not ret_keys and global_ret_keys:
                ret_keys, tau_source = global_ret_keys, "shared_retained_probe_texts"
            sem_ret = np.array([feats[k]["sem"] for k in ret_keys])
            tau = float(np.quantile(sem_ret, 0.95)) if len(sem_ret) else 1.0
            extras.setdefault(tag, {})["semantic_tau_at_5pct_false_refusal"] = tau
            extras[tag]["semantic_tau_source"] = tau_source if len(sem_ret) else "none (no retained probes; filter inactive)"
            rules = {"kw_full": lambda k: feats[k]["kw_full"], "kw_tok": lambda k: feats[k]["kw_tok"],
                     "semantic": lambda k: feats[k]["sem"] >= tau}
            for name, rule in rules.items():
                refused = {r["key"] for r in raw if r["key"] in feats and rule(r["key"])}
                stats_ = {}
                for role in ["deleted", "retained", "phantom"]:
                    ks = [r["key"] for r in raw if r["role"] == role and r["key"] in feats]
                    stats_[role] = float(np.mean([k in refused for k in ks])) if ks else None
                extras[tag][f"filter_{name}_refusal_rate"] = stats_
                tables[f"filter_{name}"] = load_table(raw, refused)
                if ret_raw:
                    tables[f"route_{name}"] = load_table(raw, refused, route=ret_raw)
        for v, T in tables.items():
            vid = f"{tag}/{v}"
            ranks = compute_ranks(T, CAN, DEFAULT, tag=vid)
            procs = subject_processes(T, CAN, DEFAULT, ranks, alphas, n_order, tag=vid)
            entry = dict(behaviour=behaviour(v))
            for role in ["deleted", "retained", "phantom"]:
                entry[role] = summarize_processes(procs, alphas, role, boot_median=True)
                keys = [k for k in ranks if T[k]["role"] == role]
                entry[role]["rank_law"] = rank_law(ranks, keys, 10)
            main[vid] = entry
            for p in procs:
                if p["role"] == "phantom" or (p["role"] == "deleted" and behaviour(v) in NULL_BEHAVIOURS):
                    all_null_procs.append(dict(p, variant=vid))
            # baselines
            tb = twin_and_binomial(T, CAN, alphas, n_order, vid)
            baselines[vid] = {role: {kname: rate_summary(tb, kname, role) for kname in tb[0] if "@" in kname}
                              for role in ["deleted", "retained", "phantom"]} if tb else {}
            # ablations on primary variants
            if v in ("planted", "npo", "ga", "retrain", "base", "rmu", "graddiff") or v.startswith("adaptive"):
                ab = {}
                grid = [("M", dict(M=M)) for M in [2, 5, 10, 20]] + \
                       [("kinds", dict(kinds=ks)) for ks in [["direct"], ["paraphrase"], ["cloze"], ["fewshot"], ["direct", "paraphrase"],
                                                             ["direct", "paraphrase", "cloze"]]] + \
                       [("strategy", dict(strategy=s)) for s in E.STRATEGIES] + \
                       [("n_canaries", dict(n_canaries=n)) for n in [1, 2, 3, 5]] + \
                       [("method", dict(method=m)) for m in ["sum", "rank"]] + [("norm", dict(norm="mean"))] + \
                       [("adaptive_kinds", dict(adaptive_kinds=True))]
                if ref_T is not None and v != "base":
                    grid.append(("ref", dict(ref=True, method="z")))
                    grid.append(("ref_sum", dict(ref=True, method="sum")))
                for gname, ov in grid:
                    cv = dict(DEFAULT, **ov)
                    if cv["M"] > min(len(e["cand_idx"]) for e in T.values()):
                        continue
                    rk = compute_ranks(T, CAN, cv, ref_T=ref_T, tag=vid) if not cv["adaptive_kinds"] else {}
                    pr = subject_processes(T, CAN, cv, rk, alphas, n_order, tag=vid)
                    lbl = f"{gname}={ov[list(ov)[0]] if not isinstance(ov[list(ov)[0]], list) else '+'.join(ov[list(ov)[0]])}"
                    ab[lbl] = {role: summarize_processes(pr, alphas, role) for role in ["deleted", "phantom", "retained"]}
                    if gname == "ref":
                        for role in ["deleted", "retained", "phantom"]:
                            ab[lbl][role]["rank_law"] = rank_law(rk, [k for k in rk if T[k]["role"] == role], 10)
                    if gname in ("M", "kinds", "strategy", "n_canaries", "method", "ref", "adaptive_kinds"):
                        all_null_procs.extend(dict(p, variant=f"{vid}|{lbl}") for p in pr
                                              if p["role"] == "phantom" or (p["role"] == "deleted" and behaviour(v) in NULL_BEHAVIOURS))
                # per planting rate: top-1 rate of deleted canaries
                by_rate = collections.defaultdict(list)
                for k, (r, M) in ranks.items():
                    if T[k]["role"] == "deleted":
                        by_rate[T[k]["rate"]].append(r == 1)
                ab["by_rate_top1"] = {str(rr): dict(top1=float(np.mean(x)), n=len(x), k=int(sum(x)),
                                                    ci95=E.clopper_pearson(int(sum(x)), len(x))) for rr, x in sorted(by_rate.items())}
                # Stage 4: planting-rate ablation as evidence, not only top-1: e-processes restricted to the canaries planted
                # with r mentions, truncated to the same number of canaries (2, the size of the smallest rate group) so the
                # rates are compared at equal n; plus the rank law (KL to uniform) of all canaries with that rate.
                ab["by_rate_eprocess"] = {}
                for rr in sorted(by_rate):
                    rk_r = {k: v for k, v in ranks.items() if T[k]["rate"] == rr}
                    pr = subject_processes(T, CAN, dict(DEFAULT, n_canaries=2), rk_r, alphas, n_order, tag=vid)
                    ab["by_rate_eprocess"][str(rr)] = dict(n_canaries_per_process=2,
                                                           deleted=summarize_processes(pr, alphas, "deleted"),
                                                           rank_law=rank_law(rk_r, [k for k in rk_r if T[k]["role"] == "deleted"], 10))
                by_type = collections.defaultdict(list)
                for k, (r, M) in ranks.items():
                    if T[k]["role"] == "deleted":
                        by_type[T[k]["type"]].append(r == 1)
                ab["by_type_top1"] = {str(tt): dict(top1=float(np.mean(x)), n=len(x)) for tt, x in sorted(by_type.items())}
                ablations[vid] = ab
        # pre-deletion memorisation buckets (from planted model) and post-deletion evidence per bucket
        if "planted" in tables:
            Tp = tables["planted"]
            adv = {}
            for (a, sl), e in Tp.items():
                if e["role"] != "deleted":
                    continue
                cols = CAN.subset(a, sl, 10)
                X = e["S"][:, [e["cand_idx"].index(c) for c in cols]]
                P = np.exp(X - X.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
                adv[(a, sl)] = float(10 * P[:, cols.index(CAN.planted(a, sl))].mean() - 1)
            if adv:
                qs = np.quantile(list(adv.values()), [1 / 3, 2 / 3])
                bucket = {k: int(np.searchsorted(qs, v, side="right")) for k, v in adv.items()}
                mem[tag] = dict(advantage_tertile_cuts=qs.tolist(), mean_advantage=float(np.mean(list(adv.values()))), per_variant={})
                for v, T in tables.items():
                    rk = compute_ranks(T, CAN, DEFAULT, tag=f"{tag}/{v}")
                    mem[tag]["per_variant"][v] = {str(b): dict(top1=float(np.mean([rk[k][0] == 1 for k in bucket if bucket[k] == b and k in rk])),
                                                                n=int(sum(1 for k in bucket if bucket[k] == b and k in rk)))
                                                  for b in range(3)}
    # m+1 ablation up to 20 on the all-candidate subset (deleted subjects, direct+cloze probes)
    m20 = {}
    for f in sorted(glob.glob(os.path.join(rd, "probes_m20", "*.jsonl"))):
        vid = os.path.basename(f)[:-6].replace("__", "/")
        T = load_table(f)
        m20[vid] = {}
        for M in [2, 5, 10, 20]:
            for strat in ["plugin", "top1_full", "mixture_top1"]:
                cv = dict(DEFAULT, M=M, strategy=strat)
                pr = subject_processes(T, CAN, cv, compute_ranks(T, CAN, cv, tag=vid + "m20"), alphas, n_order, tag=vid)
                m20[vid][f"M={M},{strat}"] = summarize_processes(pr, alphas, "deleted")
    save_json(m20, os.path.join(out, "ablation_m20.json"))
    save_json(main, os.path.join(out, "main.json"))
    save_json(ablations, os.path.join(out, "ablations.json"))
    save_json(baselines, os.path.join(out, "baselines.json"))
    save_json(mem, os.path.join(out, "memorisation.json"))
    save_json(extras, os.path.join(out, "filters.json"))
    # pooled empirical Type-I (E15)
    t1 = {}
    for al in alphas:
        dflt = [p for p in all_null_procs if "|" not in p["variant"]]
        allp = all_null_procs
        for name, ps in [("default_config", dflt), ("all_configs", allp)]:
            k = sum(p[f"rej@{al}"] for p in ps)
            t1[f"{name},alpha={al}"] = dict(rate=k / max(1, len(ps)), k=int(k), n=len(ps), ci95=E.clopper_pearson(int(k), len(ps)))
        # Stage 4: the pooled default count above includes (a) the phantoms of derived filter_/route_ variants, which are
        # copies or near-copies of the planted model's phantom processes, and (b) 3 orderings per subject. These views use
        # only model-generated (trained / never-trained) variants, optionally ordering 0 only, and give a subject-cluster
        # bootstrap CI (clusters = subject id within the run, shared by every model of the run).
        def derived(v):
            return v.split("/", 1)[1].startswith(("filter_", "route_"))
        trained = [p for p in all_null_procs if "|" not in p["variant"] and not derived(p["variant"])]
        for name, ps in [("trained_variants", trained), ("trained_variants_order0", [p for p in trained if p["order"] == 0])]:
            k = sum(p[f"rej@{al}"] for p in ps)
            t1[f"{name},alpha={al}"] = dict(rate=k / max(1, len(ps)), k=int(k), n=len(ps), ci95=E.clopper_pearson(int(k), len(ps)),
                                            n_variants=len({p["variant"] for p in ps}),
                                            subject_cluster_boot=cluster_bootstrap(ps, al),
                                            variant_subject_cluster_boot=cluster_bootstrap(ps, al, key=lambda p: (p["variant"], p["subject"])),
                                            per_variant={v: dict(k=int(sum(p[f"rej@{al}"] for p in ps if p["variant"] == v)),
                                                                 n=sum(1 for p in ps if p["variant"] == v))
                                                         for v in sorted({p["variant"] for p in ps})})
    save_json(t1, os.path.join(out, "typeI_empirical.json"))
    # MIA, traps, utility, sampled answers, detector, RAG
    mia, trap, util, samp = {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(rd, "mia", "*.jsonl"))):
        vid = os.path.basename(f)[:-6].replace("__", "/")
        m = mia_baseline(read_jsonl(f), alphas)
        mia[vid] = {role: {f"rej@{al}": rate_summary(m, f"rej@{al}", role) for al in alphas} for role in ["deleted", "retained", "phantom"]}
        for role in ["deleted", "retained", "phantom"]:
            vv = [x["mean_minkpp"] for x in m if x["role"] == role]
            mia[vid][role]["mean_minkpp"] = float(np.mean(vv)) if vv else None
    for f in sorted(glob.glob(os.path.join(rd, "trap", "*.jsonl"))):
        vid = os.path.basename(f)[:-6].replace("__", "/")
        tr = trap_baseline(read_jsonl(f), CAN, alphas)
        trap[vid] = {role: dict(top1=float(np.mean([x["rank"] == 1 for x in tr if x["role"] == role])) if any(x["role"] == role for x in tr) else None,
                                **{f"rej@{al}": rate_summary(tr, f"rej@{al}", role) for al in alphas}) for role in ["deleted", "retained", "phantom"]}
        # Stage 4: one trap per subject with M = 10 candidates can never reach e >= 1/alpha (max e = M = 10 < 20), so the
        # per-subject trap test is structurally powerless at alpha <= 0.1. Report also an auditor-level test pooled over
        # subjects (independent keys -> independent ranks under H0): plug-in e-process over subjects in id order + fixed-n binomial.
        for role in ["deleted", "retained", "phantom"]:
            rr = sorted((x for x in tr if x["role"] == role), key=lambda x: x["subject"])
            if rr and trap[vid][role] is not None:
                rks = [x["rank"] for x in rr]
                M = int(cfg["trap"]["n_candidates"])
                W = E.eprocess_path(rks, M, "plugin")
                trap[vid][role]["pooled_over_subjects"] = dict(
                    n_subjects=len(rks), logW=float(np.log(max(W[-1], 1e-300))), per_subject_max_e=M,
                    **{f"eproc_rej@{al}": E.first_crossing(W, al) is not None for al in alphas},
                    **{f"eproc_subjects_to_decision@{al}": E.first_crossing(W, al) for al in alphas},
                    **{f"binom_fixedn_rej@{al}": bool(E.fixed_n_binom_reject([int(r == 1) for r in rks], len(rks), 1 / M, al)) for al in alphas})
    for f in sorted(glob.glob(os.path.join(rd, "utility", "*.jsonl"))):
        util[os.path.basename(f)[:-6].replace("__", "/")] = utility_summary(read_jsonl(f), cfg["utility"]["tofu_retain_n"])
    for f in sorted(glob.glob(os.path.join(rd, "samples", "*.jsonl"))):
        samp[os.path.basename(f)[:-6].replace("__", "/")] = sampled_analysis(read_jsonl(f), CAN, alphas)
    save_json(dict(mia=mia, trap=trap), os.path.join(out, "baselines_mia_trap.json"))
    save_json(util, os.path.join(out, "utility.json"))
    save_json(samp, os.path.join(out, "sampled.json"))
    det = {}
    for f in sorted(glob.glob(os.path.join(rd, "detector", "*.json"))):
        if "target_sentences" not in f:
            det[os.path.basename(f)[:-5]] = load_json(f)
    save_json(det, os.path.join(out, "detector.json"))
    rag = analyze_rag(cfg, CAN, alphas, n_order)
    save_json(rag, os.path.join(out, "rag.json"))
    print(f"[analyze] done in {time.time() - t0:.0f}s -> {out}")


def analyze_rag(cfg, CAN, alphas, n_order):
    rd = os.path.join(run_dir(cfg), "rag")
    out = {}
    feats_p = os.path.join(run_dir(cfg), "filters", "rag_filter_feats.json")
    feats = load_json(feats_p) if os.path.exists(feats_p) else None
    tables = {}
    for f in sorted(glob.glob(os.path.join(rd, "probes_*.jsonl"))):
        beh = os.path.basename(f)[7:-6]
        rows = read_jsonl(f)
        tables[beh] = (rows, load_table(rows))
    if "lazy" in tables and feats:
        rows = tables["lazy"][0]
        ret = np.array([feats[r["key"]]["sem"] for r in rows if r["role"] == "retained" and r["key"] in feats])
        tau = float(np.quantile(ret, 0.95)) if len(ret) else 1.0
        for name, rule in {"kw_full": lambda k: feats[k]["kw_full"], "kw_tok": lambda k: feats[k]["kw_tok"],
                           "semantic": lambda k: feats[k]["sem"] >= tau}.items():
            refused = {r["key"] for r in rows if r["key"] in feats and rule(r["key"])}
            tables[f"filter_{name}"] = (rows, load_table(rows, refused))
    for beh, (rows, T) in tables.items():
        ranks = compute_ranks(T, CAN, DEFAULT, tag=f"rag/{beh}")
        procs = subject_processes(T, CAN, DEFAULT, ranks, alphas, n_order, tag=f"rag/{beh}")
        out[beh] = {role: summarize_processes(procs, alphas, role, boot_median=True) for role in ["deleted", "retained", "phantom"]}
        for role in ["deleted", "retained", "phantom"]:
            rr = [r for r in rows if r["role"] == role]
            if rr:
                out[beh][role]["subject_hit_rate"] = float(np.mean([r["subj_hit"] for r in rr]))
                out[beh][role]["slot_hit_rate"] = float(np.mean([r["slot_hit"] for r in rr]))
            out[beh][role]["rank_law"] = rank_law(ranks, [k for k in ranks if T[k]["role"] == role], 10)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only_mc", action="store_true")
    a = ap.parse_args()
    analyze(load_config(a.config), a.only_mc)
