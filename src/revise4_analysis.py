"""Round-4 review fixes (CPU only) -> results/revise4/<part>.json. Re-analyses of existing probe logs; no model is loaded.

python src/revise4_analysis.py [part ...] [--force]
Parts:
  grinding_pooled  key grinding: exact re-runs of results/revise1/grinding.json (20 repetitions, rng seed 1) and of
                   results/revise3/grinding_cluster.json (50 repetitions, rng seed 11) that keep the per-(repetition, subject)
                   outcomes (both reproductions are checked against the stored files), then both runs pooled for every G
                   they share (1, 10, 100, 1,000) with subject-cluster bootstrap intervals
  routing_pooled   semantic routing to the compliant retrain (results/revise3/routing_rec.json): accusations of the three
                   reported verifiers summed over the four replicates (keys 0, 1, 2 of Qwen2.5-3B and key-0 Llama-3.2-3B),
                   the largest per-replicate count of any of them, and the default's range under semantic refusal
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import load_json, save_json, stable_int
from fmp import eprocess as E
import rev1lib as L
import revise1_analysis as R1
import revise2_analysis as R2

OUT = os.path.join(L.ROOT, "results", "revise4")
ALPHA = 0.05
TARGETS = ["qwen3b__retrain", "llama3b__retrain"]


def save(part, obj):
    os.makedirs(OUT, exist_ok=True)
    save_json(obj, os.path.join(OUT, f"{part}.json"))
    print(f"[revise4] wrote {part}.json", flush=True)


# ------------------------------------------------------------------------------------------------ grinding_pooled
def _rank_table(Vx, a):
    """Rank of position p of the planted value for each fixed-order canary (identical to revise1/revise2/revise3)."""
    T = []
    for s in R1.ord0(Vx, a):
        sc = L.zagg(Vx.X[(a, s)]["S"])
        T.append([L.tie_rank(sc, p, stable_int("grind", Vx.vid, a, s, p)) for p in range(10)])
    return np.array(T)


def _grind(rng, tabs, G, reps):
    """One grinding run: G keys tried against the never-trained model, best kept, compliant retrain audited.
    Same draws and loop order as revise1_analysis.part_grinding and revise3_analysis.part_grinding_cluster."""
    items = []
    for rep in range(reps):
        for a, (Rb, Rt) in tabs.items():
            P = rng.integers(0, 10, size=(G, Rb.shape[0]))
            Wb = E.eprocess_paths_vec(Rb[np.arange(Rb.shape[0])[None, :], P], 10, "plugin")
            best = int(np.argmax(Wb.max(1)))
            rt = Rt[np.arange(Rt.shape[0]), P[best]]
            items.append(dict(rep=rep, subject=a, acc=L.first_cross(E.eprocess_path(list(rt), 10, "plugin"), ALPHA) is not None))
    return items


def _summ(items):
    k = int(sum(p["acc"] for p in items)); n = len(items)
    return dict(k=k, n=n, rate=k / n, ci95_iid=E.clopper_pearson(k, n),
                cluster_subject=R2.cl_boot(items, lambda p: p["subject"], "acc"))


def part_grinding_pooled():
    Vb = R1.V("key0", "qwen3b__base")
    tabs = {}
    for t in TARGETS:
        Vt = R1.V("key0", t)
        subs = [a for a in Vb.subjects("deleted") if a in Vt.subjects("deleted")]
        tabs[t] = {a: (_rank_table(Vb, a), _rank_table(Vt, a)) for a in subs}
    # run 1: results/revise1/grinding.json (rng seed 1; targets in turn, G = 1, 10, 100, 1000, 10000, 20 repetitions)
    rng1 = np.random.default_rng(1)
    run1 = {t: {} for t in TARGETS}
    for t in TARGETS:
        for G in [1, 10, 100, 1000, 10000]:
            run1[t][G] = _grind(rng1, tabs[t], G, 20)
    # run 2: results/revise3/grinding_cluster.json (rng seed 11; G = 1, 2, 5, 10 for both targets, then G = 100, 1000)
    rng2 = np.random.default_rng(11)
    run2 = {t: {} for t in TARGETS}
    for t in TARGETS:
        for G in [1, 2, 5, 10]:
            run2[t][G] = _grind(rng2, tabs[t], G, 50)
    for t in TARGETS:
        for G in [100, 1000]:
            run2[t][G] = _grind(rng2, tabs[t], G, 50)
    g1 = load_json(os.path.join(L.ROOT, "results/revise1/grinding.json"))
    g3 = load_json(os.path.join(L.ROOT, "results/revise3/grinding_cluster.json"))
    ok1 = all(abs(np.mean([p["acc"] for p in run1[t][G]]) - g1[t][str(G)]["false_accusation_target"]) < 1e-12
              for t in TARGETS for G in [1, 10, 100, 1000, 10000])
    ok3 = all(sum(p["acc"] for p in run2[t][G]) == g3[t][str(G)]["k"]
              and _summ(run2[t][G])["cluster_subject"]["ci95"] == g3[t][str(G)]["cluster_subject"]["ci95"]
              for t in TARGETS for G in [1, 2, 5, 10, 100, 1000])
    assert ok1, "re-run does not reproduce results/revise1/grinding.json"
    assert ok3, "re-run does not reproduce results/revise3/grinding_cluster.json"
    res = {}
    for t in TARGETS:
        res[t] = {}
        for G in [1, 10, 100, 1000]:
            res[t][str(G)] = dict(run1=_summ(run1[t][G]), run2=_summ(run2[t][G]), pooled=_summ(run1[t][G] + run2[t][G]))
    res["reproduces"] = {"results/revise1/grinding.json": ok1, "results/revise3/grinding_cluster.json": ok3}
    res["_notes"] = ("run1 = results/revise1/grinding.json (20 repetitions x 20 deleted subjects, rng seed 1), run2 = "
                     "results/revise3/grinding_cluster.json (50 repetitions x 20 subjects, rng seed 11), both re-run exactly with "
                     "per-(repetition, subject) outcomes kept. Same procedure: G keys tried against the never-trained Qwen2.5-3B, "
                     "the key with the largest wealth there kept, the compliant retrain audited (fixed order, default plug-in "
                     "verifier, alpha = 0.05). pooled = both runs (70 repetitions x 20 subjects); cluster_subject = bootstrap over "
                     "the 20 deleted subjects (revise2_analysis.cl_boot, 4,000 resamples, seed 0); ci95_iid = Clopper-Pearson, "
                     "which ignores that repetitions reuse the subjects' rank tables.")
    save("grinding_pooled", res)


# ------------------------------------------------------------------------------------------------ routing_pooled
def part_routing_pooled():
    r = load_json(os.path.join(L.ROOT, "results/revise3/routing_rec.json"))
    reps = ["key0/qwen3b", "key1/qwen3b", "key2/qwen3b", "key0/llama3b"]
    vers = ["default", "kind_mix", "km_prior_0.75"]
    res = {}
    for b in ["route_retrain_semantic", "compliant", "refuse_semantic"]:
        res[b] = {}
        for v in vers:
            ks = [r[m][b][v]["k"] for m in reps]; ns = [r[m][b][v]["n"] for m in reps]
            res[b][v] = dict(per_replicate={m: k for m, k in zip(reps, ks)}, k=int(sum(ks)), n=int(sum(ns)), min=int(min(ks)), max=int(max(ks)))
    sem = [r[m][b][v]["k"] for m in reps for v in vers for b in ["route_retrain_semantic"]]
    sem += [r["key0/qwen3b"]["route_base_semantic"][v]["k"] for v in vers]
    res["semantic_routing_max_any_reported_verifier"] = int(max(sem))
    res["mcnemar_km_prior_0.75_per_replicate"] = {m: r[m]["route_retrain_semantic"]["mcnemar_vs_compliant"]["km_prior_0.75"] for m in reps}
    res["_notes"] = ("Sums over the four replicates of results/revise3/routing_rec.json (ordering 0, 20 deleted subjects each) for the "
                     "default, the kind mixture and the recommended KM-prior (lambda = 0.75). The replicates are not independent "
                     "(key-0 Qwen2.5-3B and key-0 Llama-3.2-3B share subjects and canaries), so no pooled test is computed. "
                     "semantic_routing_max_any_reported_verifier = largest count of these three verifiers on any replicate, with "
                     "semantic matches routed to the compliant retrain or (key-0 Qwen) to the never-trained base model.")
    save("routing_pooled", res)


PARTS = ["grinding_pooled", "routing_pooled"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in a.parts or PARTS:
        if not a.force and os.path.exists(os.path.join(OUT, f"{p}.json")):
            print(f"[revise4] skip {p} (exists)"); continue
        t = time.time()
        globals()[f"part_{p}"]()
        print(f"[revise4] {p} took {time.time() - t:.0f}s", flush=True)
