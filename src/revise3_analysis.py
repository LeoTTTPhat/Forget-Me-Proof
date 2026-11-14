"""Round-3 audit fixes (CPU only) -> results/revise3/<part>.json. Re-analyses of existing probe logs; no model is loaded.

python src/revise3_analysis.py [part ...] [--force]
Parts:
  routing_rec    recommended KM-prior verifier (lambda = 0.75) on every suppressive provider of Table S-filters
                 (full-name, semantic and any-name-token filters; refuse / route / random scores), plus exact McNemar
                 tests against the compliant retrain on the same subjects
  attribute_rec  five-kind mixtures with the attribute probes, KM-prior with lambda = 0.75 (the recommended value) and
                 lambda = 0.5 (reproduces results/revise2/attribute.json), under the any-name-token router
  grinding_cluster  key grinding (same procedure and random stream as results/revise2/grinding2.json) with
                 subject-cluster bootstrap intervals; pooled G = 10 rate over revision 1 and revision 2 runs
  certificate_rec  Theorem 2(v)/(vi) certificates for the recommended verifier (per-kind fixed bets, lambda = 0.75,
                 threshold |T|/alpha), under each kind's pooled i.i.d. rank law on key-0 lazy
"""
import argparse, collections, math, os, sys, time
import numpy as np
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import read_jsonl, load_json, save_json, stable_int
from fmp import eprocess as E
from fmp.analyze import ordering
import rev1lib as L
import revise1_analysis as R1
import revise2_analysis as R2

OUT = os.path.join(L.ROOT, "results", "revise3")
ALPHA = 0.05
M = 10
LAM_REC = 0.75


def save(part, obj):
    os.makedirs(OUT, exist_ok=True)
    save_json(obj, os.path.join(OUT, f"{part}.json"))
    print(f"[revise3] wrote {part}.json", flush=True)


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test on paired booleans (a: provider under test, b: compliant)."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return dict(n10=0, n01=0, p_two_sided=1.0)
    p = min(1.0, 2 * st.binom.cdf(min(n10, n01), n, 0.5))
    return dict(n10=n10, n01=n01, p_two_sided=float(p))


# ------------------------------------------------------------------------------------------------ routing_rec
VERS = ["default", "kind_mix", "km_prior_0.5", "km_prior_0.75", "km_prior_1.0"]


def part_routing_rec():
    setups = [("key0", "qwen3b", ["retrain", "base"]), ("key0", "llama3b", ["retrain"]), ("key1", "qwen3b", ["retrain"]),
              ("key2", "qwen3b", ["retrain"])]
    res = {}
    for key, tag, routes in setups:
        beh = {"lazy": R1.V(key, f"{tag}__planted"), "compliant": R1.V(key, f"{tag}__retrain")}
        for rule in ["kw_full", "semantic", "kw_tok"]:
            # identical construction (and hence tie-break seeds) to results/revise1/routing.json and revise2/recommended.json
            beh[f"refuse_{rule}"] = R1.V(key, f"{tag}__filter_{rule}", rows=R1.derived_rows(key, tag, rule))
            for r_ in routes:
                beh[f"route_{r_}_{rule}"] = R1.V(key, f"{tag}__route_{rule}" + ("" if r_ == "retrain" else f"_{r_}"),
                                                 rows=R1.derived_rows(key, tag, rule, route_from=f"{tag}__{r_}"))
            beh[f"noise_{rule}"] = R1.V(key, f"{tag}__noise_{rule}", rows=R1.derived_rows(key, tag, rule, noise=True))
        acc = {}
        subs = beh["lazy"].subjects("deleted")
        for b, Vx in beh.items():
            acc[b] = {}
            for a in subs:
                if a not in Vx.subjects("deleted"):
                    continue
                v = R2.verifiers(Vx, a, 0)
                acc[b][a] = {x: v[x] is not None for x in VERS}
        out = {}
        for b in beh:
            d = {x: L.rate(sum(acc[b][a][x] for a in acc[b]), len(acc[b])) for x in VERS}
            common = [a for a in subs if a in acc[b] and a in acc["compliant"]]
            d["mcnemar_vs_compliant"] = {x: mcnemar_exact([acc[b][a][x] for a in common], [acc["compliant"][a][x] for a in common])
                                         for x in ("default", "kind_mix", "km_prior_0.75")}
            out[b] = d
        res[f"{key}/{tag}"] = out
    res["_notes"] = ("Ordering 0, deleted subjects, alpha = 0.05. Derived providers built exactly as in results/revise1/routing.json "
                     "(refuse = constant row; route_<m> = filtered probes answered by model m's logged scores; noise = pi-independent "
                     "N(0, 10^2) scores). km_prior_<lam> = average over the 4 probe kinds of fixed prior-shaped e-processes "
                     "(Sec. III-D); the recommended verifier is km_prior_0.75. McNemar: exact two-sided test against the compliant "
                     "retrain on the same 20 subjects.")
    save("routing_rec", res)


# ------------------------------------------------------------------------------------------------ attribute_rec
def part_attribute_rec():
    det = {}
    for key, name, grp in R2.ATTR_VARIANTS:
        f = os.path.join(L.ROOT, "results/rev2_attr/probes", f"{name}.jsonl")
        if not os.path.exists(f):
            det[name] = "missing"; continue
        Va = L.load_variant(key, name, rows=read_jsonl(f)); Va.vid = name.replace("__", "/") + "@attr"
        mf = os.path.join(L.ROOT, "results/rev1/probes" if "scrub" in name else L.RUNDIR[key] + "/probes", f"{name}.jsonl")
        Vm = R1.V(key, name, rows=read_jsonl(mf)) if "scrub" in name else R1.V(key, name)
        d = {}
        for role in ("deleted", "phantom"):
            subs = list(Va.subjects(role))
            if not subs:
                continue
            acc = collections.defaultdict(list)
            for a in subs:
                for o in range(3 if role == "phantom" else 1):
                    sl = ordering(Va.slots(a), a, o)
                    ra = [R2.attr_rank(Va, a, s) for s in sl]
                    for lam in (0.5, LAM_REC):
                        f_ = (1 - lam) + lam * M * R2.Q
                        acc[f"attribute_only_prior{lam}"].append(L.first_cross(np.r_[1.0, np.cumprod(f_[np.array(ra) - 1])], ALPHA) is not None)
                    if role == "phantom" or a not in Vm.subjects(role) or len(Vm.slots(a)) != len(sl):
                        continue
                    ks = [dict(L.kind_ranks(Vm, a, s), attribute=r_) for s, r_ in zip(sl, ra)]
                    for lam in (0.5, LAM_REC):
                        acc[f"km_prior5_{lam}"].append(L.first_cross(R2.km_prior_path(ks, lam, R2.KINDS5), ALPHA) is not None)
                        acc[f"km_prior4_{lam}"].append(L.first_cross(R2.km_prior_path(ks, lam, L.KINDS), ALPHA) is not None)
                    rk_ref = [dict({k: L.tie_rank(np.zeros(M), Vm.X[(a, s)]["p"], stable_int("ref", a, s, k)) for k in L.KINDS}, attribute=r_)
                              for s, r_ in zip(sl, ra)]
                    for lam in (0.5, LAM_REC):
                        acc[f"anytok_refuse_km_prior5_{lam}"].append(L.first_cross(R2.km_prior_path(rk_ref, lam, R2.KINDS5), ALPHA) is not None)
                    acc["anytok_refuse_kind_mix5"].append(L.first_cross(R2.km_plugin_path(rk_ref, R2.KINDS5), ALPHA) is not None)
                    if key == "key0" and name.endswith("__planted"):
                        tag = name.split("__")[0]
                        Vc = R1.V(key, f"{tag}__retrain")
                        if a in Vc.subjects("deleted"):
                            rk_rt = [dict(L.kind_ranks(Vc, a, s), attribute=r_) for s, r_ in zip(sl, ra)]
                            for lam in (0.5, LAM_REC):
                                acc[f"anytok_route_km_prior5_{lam}"].append(L.first_cross(R2.km_prior_path(rk_rt, lam, R2.KINDS5), ALPHA) is not None)
                            acc["anytok_route_kind_mix5"].append(L.first_cross(R2.km_plugin_path(rk_rt, R2.KINDS5), ALPHA) is not None)
            d[role] = {k: L.rate(sum(v), len(v)) for k, v in acc.items()}
        det[name] = dict(group=grp, **d)
    ph = collections.Counter()
    for name, d in det.items():
        if isinstance(d, dict) and "phantom" in d:
            for k, v in d["phantom"].items():
                ph[(k, "k")] += v["k"]; ph[(k, "n")] += v["n"]
    res = dict(detection=det,
               phantom_pooled={k: L.rate(ph[(k, "k")], ph[(k, "n")]) for k in {kk for kk, _ in ph}},
               _notes=("Same probes, ranks and tie-breaks as results/revise2/attribute.json. km_prior5_<lam> = five-kind KM-prior "
                       "(four name-bearing kinds + attribute kind), fixed prior-shaped bets f = (1-lam) + lam M q; lam = 0.75 is the "
                       "recommended value, lam = 0.5 reproduces the 'km_prior5' entries of attribute.json. anytok_refuse: the 16 "
                       "name-bearing probes are refused (uniform random ranks); anytok_route: they are answered by the compliant retrain "
                       "(key 0, planted models only). Deleted subjects: ordering 0; phantoms: 3 orderings, attribute-only tests."))
    save("attribute_rec", res)


# ------------------------------------------------------------------------------------------------ grinding_cluster
def part_grinding_cluster():
    """Exact re-run of revise2.part_grinding2 (same rng seed and loop order), keeping per-(repetition, subject) outcomes."""
    Vb = R1.V("key0", "qwen3b__base")
    rng = np.random.default_rng(11)
    res = {}

    def rank_table(Vx, a):
        T = []
        for s in R1.ord0(Vx, a):
            sc = L.zagg(Vx.X[(a, s)]["S"])
            T.append([L.tie_rank(sc, p, stable_int("grind", Vx.vid, a, s, p)) for p in range(10)])
        return np.array(T)

    targets = ["qwen3b__retrain", "llama3b__retrain"]
    TABS = {}
    for target in targets:
        Vt = R1.V("key0", target)
        subs = [a for a in Vb.subjects("deleted") if a in Vt.subjects("deleted")]
        TABS[target] = {a: (rank_table(Vb, a), rank_table(Vt, a)) for a in subs}
        res[target] = {}

    def run(target, G):
        items = []
        for rep in range(50):
            for a, (Rb, Rt) in TABS[target].items():
                P = rng.integers(0, 10, size=(G, Rb.shape[0]))
                Wb = E.eprocess_paths_vec(Rb[np.arange(Rb.shape[0])[None, :], P], 10, "plugin")
                best = int(np.argmax(Wb.max(1)))
                rt = Rt[np.arange(Rt.shape[0]), P[best]]
                items.append(dict(rep=rep, subject=a, acc=L.first_cross(E.eprocess_path(list(rt), 10, "plugin"), ALPHA) is not None))
        k = sum(p["acc"] for p in items)
        res[target][str(G)] = dict(k=int(k), n=len(items), false_accusation_target=k / len(items),
                                   ci95_iid=E.clopper_pearson(int(k), len(items)),
                                   cluster_subject=R2.cl_boot(items, lambda p: p["subject"], "acc"))
    # phase 1: the exact random stream and loop order of results/revise2/grinding2.json (reproduced, checked below)
    for target in targets:
        for G in [1, 2, 5, 10]:
            run(target, G)
    # phase 2: 100 and 1,000 keys continue the same stream (new draws, not a reproduction of results/revise1/grinding.json)
    for target in targets:
        for G in [100, 1000]:
            run(target, G)
    g2 = load_json(os.path.join(L.ROOT, "results/revise2/grinding2.json"))
    g1 = load_json(os.path.join(L.ROOT, "results/revise1/grinding.json"))
    res["reproduces_grinding2"] = all(abs(res[t][G]["false_accusation_target"] - g2[t][G]["false_accusation_target"]) < 1e-12
                                      for t in ("qwen3b__retrain", "llama3b__retrain") for G in ("1", "2", "5", "10"))
    pooled = {}
    for t in ("qwen3b__retrain", "llama3b__retrain"):
        for G in ("1", "10", "100", "1000"):
            k1 = round(g1[t][G]["false_accusation_target"] * g1[t][G]["n"]); n1 = g1[t][G]["n"]
            k2 = res[t][G]["k"]; n2 = res[t][G]["n"]
            pooled[f"{t}/{G}"] = dict(rev1=L.rate(int(k1), n1), rev2=L.rate(int(k2), n2), pooled=L.rate(int(k1 + k2), n1 + n2))
    res["pooled_rev1_rev2"] = pooled
    res["_notes"] = ("Same procedure, seed and loop order as results/revise2/grinding2.json (G keys tried against the never-trained "
                     "Qwen2.5-3B, best kept, compliant retrain audited, fixed order, 50 repetitions x 20 subjects). cluster_subject: "
                     "bootstrap over the 20 deleted subjects (each repetition reuses the same subjects' rank tables). pooled_rev1_rev2: "
                     "revision-1 run (20 repetitions, different random stream) and this run pooled; the difference between the two "
                     "runs at G = 10 shows the Monte Carlo spread. G = 100 and 1,000 continue the same random stream after G = 10 (new draws, "
                     "not a reproduction of results/revise1/grinding.json).")
    save("grinding_cluster", res)


# ------------------------------------------------------------------------------------------------ certificate_rec
def part_certificate_rec():
    res = {}
    f = (1 - LAM_REC) + LAM_REC * M * R2.Q
    T = len(L.KINDS)
    for key, name in [("key0", "qwen3b__planted"), ("key0", "qwen3b__npo"), ("key0", "llama3b__planted"), ("key0", "qwen7b__planted")]:
        Vx = R1.V(key, name)
        per = {}
        for k in L.KINDS:
            R = []
            for a in Vx.subjects("deleted"):
                R += [kr[k] for kr in R1.kind_seq(Vx, a, 0)]
            pooled = np.bincount(R, minlength=M + 1)[1:] + 0.5; pooled = pooled / pooled.sum()
            g_ = float((pooled * np.log(f)).sum()); c_ = float(np.log(f.max() / f.min()))
            cert = R2.azuma_B(pooled, lambda eta, g_=g_, c_=c_: (g_, c_), alpha=ALPHA / T)
            per[k] = dict(D=float(E.kl(pooled, np.full(M, 1 / M))), top1=float(pooled[0]), **cert)
        Bs = [v["B"] for v in per.values() if v["B"] is not None]
        res[f"{key}/{name}"] = dict(per_kind=per, B_mixture=min(Bs) if Bs else None,
                                    E_tau_bound_mixture=min(v["E_tau_bound"] for v in per.values()),
                                    best_kind=min(per, key=lambda k: per[k]["E_tau_bound"]))
    res["_notes"] = ("Recommended verifier = average of 4 kind-specific fixed prior-shaped e-processes (lambda = 0.75). Its wealth is at "
                     "least max_t W^t / 4 (Prop. 3), so it crosses 1/alpha no later than any kind's e-process crosses 4/alpha. Each "
                     "kind's certificate is Thm 2(v)/(vi) for a fixed bet (no regret term) with threshold log(4/alpha), gamma = "
                     "E_P[log f(R)] and c = log(max f / min f), under that kind's pooled i.i.d. rank law (deleted canaries, ordering 0, "
                     "add-0.5 smoothing). B_mixture / E_tau_bound_mixture = minimum over kinds (valid because tau_mix <= tau_t(4/alpha) "
                     "pathwise for every t).")
    save("certificate_rec", res)


PARTS = ["routing_rec", "attribute_rec", "grinding_cluster", "certificate_rec"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in a.parts or PARTS:
        if not a.force and os.path.exists(os.path.join(OUT, f"{p}.json")):
            print(f"[revise3] skip {p} (exists)"); continue
        t = time.time()
        globals()[f"part_{p}"]()
        print(f"[revise3] {p} took {time.time() - t:.0f}s", flush=True)
