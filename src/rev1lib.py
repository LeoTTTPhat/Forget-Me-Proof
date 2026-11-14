"""Stage 7 (revision 1) analysis library, CPU only. Loads the per-probe score logs once into compact per-canary arrays
and re-implements the verifier so that new aggregators, bets, baselines and attacks can be evaluated on exactly the
same logs as the paper. Tie-breaking seeds match src/fmp/analyze.py, so the default configuration reproduces
results/*/analysis/main.json (checked by rev1_check()).
"""
import collections, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import load_config, read_jsonl, stable_int, RESULTS, load_json
from fmp import eprocess as E
from fmp.analyze import Canaries, ordering

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"key0": "src/configs/base.json", "key1": "src/configs/key1.json", "key2": "src/configs/key2.json"}
RUNDIR = {"key0": "results/main", "key1": "results/key1", "key2": "results/key2"}
KINDS = ["direct", "paraphrase", "cloze", "fewshot"]
# probes whose text never contains the subject's full name (paraphrases refer to "the author <Surname>" / "<I>. <Surname>")
NAME_FREE = {"paraphrase0", "paraphrase1", "paraphrase2", "paraphrase3", "fewshot1", "fewshot3"}
# probes whose prompt/stem was never used for planting (cloze0/1 use stems 2,3; paraphrases and few-shot 1,3 use pq templates)
UNSEEN = {"paraphrase0", "paraphrase1", "paraphrase2", "paraphrase3", "cloze0", "cloze1", "fewshot1", "fewshot3"}
M10 = 10

_CAN = {}


def canaries(key):
    if key not in _CAN:
        _CAN[key] = Canaries(load_config(os.path.join(ROOT, CFG[key])))
    return _CAN[key]


class Variant:
    """Compact per-canary arrays for one probe log. X[(a, sl)] = dict(S: (P, 10) summed log-probs in scored-subset
    column order, pids, kinds, refused (P,), planted position p, role, rate)."""

    def __init__(self, key, rows, vid, M=M10):
        self.key, self.vid, self.M = key, vid, M
        CAN = canaries(key)
        g = collections.defaultdict(list)
        for r in rows:
            if r.get("slot") == "trap":
                continue
            g[(r["subject"], r["slot"])].append(r)
        self.X = {}
        for (a, sl), rs in g.items():
            rs = sorted(rs, key=lambda r: r["pid"])
            cols = CAN.subset(a, sl, M)
            ci = rs[0]["cand_idx"]
            if not set(cols) <= set(ci):
                continue
            pos = [ci.index(c) for c in cols]
            S = np.array([[r["scores"][i][0] for i in pos] for r in rs], float)
            ref = np.array([bool(r.get("refused")) for r in rs])
            S[ref] = 0.0
            self.X[(a, sl)] = dict(S=S, pids=[r["pid"] for r in rs], kinds=[r["kind"] for r in rs], refused=ref,
                                   p=cols.index(CAN.planted(a, sl)), role=rs[0]["role"], rate=rs[0].get("rate"))

    def subjects(self, role=None):
        return sorted({a for (a, sl), e in self.X.items() if role is None or e["role"] == role})

    def slots(self, a):
        return sorted(sl for (s, sl) in self.X if s == a)


def load_variant(key, name, sub="probes", rows=None):
    """name = '<tag>__<variant>' (e.g. qwen3b__planted)."""
    if rows is None:
        rows = read_jsonl(os.path.join(ROOT, RUNDIR[key], sub, f"{name}.jsonl"))
    return Variant(key, rows, name.replace("__", "/"))


def zagg(S, sel=None):
    X = S if sel is None else S[sel]
    if len(X) == 0:
        return np.zeros(S.shape[1])
    sd = X.std(1, keepdims=True)
    Z = np.where(sd > 1e-9, (X - X.mean(1, keepdims=True)) / np.maximum(sd, 1e-9), 0.0)
    return Z.mean(0)


def tie_rank(scores, p, seed):
    return E.planted_rank(scores, p, np.random.default_rng(seed % (2 ** 32)))


def canary_rank(V, a, sl, pids=None, tagx=""):
    """Rank of the planted candidate under z-aggregation over the probes in `pids` (None = all)."""
    e = V.X[(a, sl)]
    sel = None if pids is None else np.array([i for i, q in enumerate(e["pids"]) if q in pids], int)
    s = zagg(e["S"], sel)
    seed = stable_int("tie", V.vid, a, sl, V.M) if not tagx else stable_int("tie", V.vid, a, sl, V.M, tagx)
    return tie_rank(s, e["p"], seed)


def kind_ranks(V, a, sl, tagx="kind"):
    e = V.X[(a, sl)]
    out = {}
    for k in KINDS:
        sel = np.array([i for i, q in enumerate(e["kinds"]) if q == k], int)
        s = zagg(e["S"], sel)
        out[k] = tie_rank(s, e["p"], stable_int("tie", V.vid, a, sl, V.M, tagx, k))
    return out


def plugin_path(ranks, M=M10, a0=2.0, ratio=0.5):
    return E.eprocess_path(ranks, M, "plugin", a0=a0, ratio=ratio)


def kindmix_path(kr_seq, M=M10, mode="mix"):
    """Routing-robust aggregators (valid: each kind's rank is uniform under H0 given the past, Lemma 1).
    mode='avg': per-canary e-value = average over the 4 probe kinds of each kind's own plug-in bet;
    mode='mix': average of the 4 kind-specific plug-in e-processes (a mixture of test martingales)."""
    bets = {k: E.PluginBet(M) for k in KINDS}
    W = [1.0]
    Wk = {k: 1.0 for k in KINDS}
    for kr in kr_seq:
        es = {k: bets[k].e(kr[k]) for k in KINDS}
        if mode == "avg":
            W.append(W[-1] * float(np.mean(list(es.values()))))
        else:
            for k in KINDS:
                Wk[k] *= es[k]
            W.append(float(np.mean(list(Wk.values()))))
        for k in KINDS:
            bets[k].update(kr[k])
    return np.array(W)


def first_cross(W, alpha):
    return E.first_crossing(np.asarray(W), alpha)


def subject_rank_seq(V, a, order=0, pids=None, tagx=""):
    sl = ordering(V.slots(a), a, order)
    return [canary_rank(V, a, s, pids, tagx) for s in sl], sl


def rate(k, n):
    lo, hi = E.clopper_pearson(k, n)
    return dict(k=int(k), n=int(n), rate=k / n if n else None, ci95=[lo, hi])


def detect(V, role="deleted", alpha=0.05, order=0, pids=None, strategy="plugin", lam=0.5, tagx=""):
    """Ordering-0 detection count for the default verifier (or a probe subset / alternative bet)."""
    k = n = 0
    taus = []
    for a in V.subjects(role):
        rk, _ = subject_rank_seq(V, a, order, pids, tagx)
        W = E.eprocess_path(rk, V.M, strategy, lam=lam)
        t = first_cross(W, alpha)
        k += t is not None; n += 1
        taus.append(t)
    return k, n, taus


def rev1_check():
    """Reproduce ordering-0 counts of main.json for a few variants (sanity check of the re-implementation)."""
    out = {}
    for key, name in [("key0", "qwen3b__planted"), ("key0", "qwen3b__retrain"), ("key0", "qwen3b__npo"), ("key1", "qwen3b__planted"),
                      ("key0", "llama3b__planted"), ("key2", "qwen3b__npo")]:
        V = load_variant(key, name)
        k, n, _ = detect(V)
        mj = load_json(os.path.join(ROOT, RUNDIR[key], "analysis", "main.json"))[name.replace("__", "/")]["deleted"]["alpha=0.05"]["order0"]
        out[f"{key}/{name}"] = dict(ours=f"{k}/{n}", main_json=f"{mj['k']}/{mj['n']}")
    return out


if __name__ == "__main__":
    print(json.dumps(rev1_check(), indent=1))
