"""Aggregate results/<run>/analysis/*.json of every run into results/summary.json (or results/<run>/summary.json
with --run). Every number reported in the paper must come from this file or the analysis JSONs it cites.

python src/aggregate.py                # all runs except smoke -> results/summary.json
python src/aggregate.py --run smoke    # -> results/smoke/summary.json
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import RESULTS, load_json, save_json, read_jsonl


def load(p):
    return load_json(p) if os.path.exists(p) else None


def headline_rows(main, util, alpha="0.05"):
    rows = []
    for vid, e in sorted(main.items()):
        d, ph, rt = e.get("deleted", {}), e.get("phantom", {}), e.get("retained", {})
        a = f"alpha={alpha}"
        row = dict(variant=vid, behaviour=e.get("behaviour"),
                   deleted_reject_rate=d.get(a, {}).get("reject_rate"), deleted_ci95=d.get(a, {}).get("ci95"),
                   deleted_reject_std_over_orderings=d.get(a, {}).get("reject_rate_order_std"),
                   deleted_canaries_to_decision=d.get(a, {}).get("canaries_to_decision_median"),
                   deleted_probes_to_decision=d.get(a, {}).get("probes_to_decision_median"),
                   deleted_top1=d.get("mean_top1"),
                   retained_reject_rate=rt.get(a, {}).get("reject_rate"),
                   phantom_reject_rate=ph.get(a, {}).get("reject_rate"), phantom_ci95=ph.get(a, {}).get("ci95"),
                   n_deleted_processes=d.get("n_processes"), n_phantom_processes=ph.get("n_processes"),
                   n_deleted_subjects=d.get("n_subjects"),
                   # Stage 4 independent-unit views (orderings of one subject share its canary ranks)
                   deleted_order0_k=(d.get(a, {}).get("order0") or {}).get("k"),
                   deleted_order0_n=(d.get(a, {}).get("order0") or {}).get("n"),
                   deleted_order0_ci95=(d.get(a, {}).get("order0") or {}).get("ci95"),
                   deleted_ci95_subject_cluster=(d.get(a, {}).get("subject_cluster_boot") or {}).get("reject_rate_ci95"),
                   deleted_canaries_to_decision_ci95=(d.get(a, {}).get("subject_cluster_boot") or {}).get("canaries_to_decision_median_ci95"),
                   phantom_order0_k=(ph.get(a, {}).get("order0") or {}).get("k"),
                   phantom_order0_n=(ph.get(a, {}).get("order0") or {}).get("n"),
                   phantom_ci95_subject_cluster=(ph.get(a, {}).get("subject_cluster_boot") or {}).get("reject_rate_ci95"))
        if d.get("rank_law"):
            row["deleted_kl_to_uniform"] = d["rank_law"]["kl_to_uniform"]
        uv = vid.split("/")[0] + "/planted" if ("/filter_" in vid or "/route_" in vid) else vid
        u = (util or {}).get(uv)
        if u:
            row.update({f"util_{k}": v["acc"] for k, v in u.items()})
        rows.append(row)
    return rows


def summarize_run(rd):
    an = os.path.join(rd, "analysis")
    if not os.path.isdir(an):
        return None
    main = load(os.path.join(an, "main.json")) or {}
    util = load(os.path.join(an, "utility.json")) or {}
    out = dict(headline=headline_rows(main, util),
               typeI_empirical=load(os.path.join(an, "typeI_empirical.json")),
               mc_null=load(os.path.join(an, "mc_null.json")),
               baselines=load(os.path.join(an, "baselines.json")),
               baselines_mia_trap=load(os.path.join(an, "baselines_mia_trap.json")),
               ablations=load(os.path.join(an, "ablations.json")),
               ablation_m20=load(os.path.join(an, "ablation_m20.json")),
               memorisation=load(os.path.join(an, "memorisation.json")),
               filters=load(os.path.join(an, "filters.json")),
               utility=util, sampled=load(os.path.join(an, "sampled.json")),
               detector=load(os.path.join(an, "detector.json")), rag=load(os.path.join(an, "rag.json")))
    tim = read_jsonl(os.path.join(rd, "timings.jsonl"))
    out["timings_hours"] = {}
    for t in tim:
        out["timings_hours"][t["step"]] = round(out["timings_hours"].get(t["step"], 0) + t["seconds"] / 3600, 4)
    out["total_compute_hours_logged"] = round(sum(v for k, v in out["timings_hours"].items() if not k.startswith("evaluate")), 3)
    return out


# ------------------------------------------------------------------------------------------------- Stage 3 additions
# Seeds: the "main" run is key0 / fine-tune seed 0, the "key1" run is key1 / fine-tune seed 1 (src/configs/key1.json).
# Within a run, each subject's e-process is evaluated over 3 canary orderings (reject_rate_order_mean/std).
SEED_OF_RUN = {"main": "key0/seed0", "key1": "key1/seed1", "key2": "key2/seed2"}


def t_ci(vals, conf=0.95, bounded=False):
    import math
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return dict(mean=m, std=None, n=1, ci95=None)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    from scipy import stats
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * sd / math.sqrt(n)
    lo, hi = m - h, m + h
    if bounded:  # Stage 4: rates/accuracies live in [0, 1]; report the t-interval clipped to that range (flagged)
        lo, hi = max(0.0, lo), min(1.0, hi)
    return dict(mean=m, std=sd, n=n, ci95=[lo, hi], **({"ci95_clipped_to_unit": True} if bounded and (lo != m - h or hi != m + h) else {}))


RATE_METRICS = {"deleted_reject_rate", "deleted_top1", "retained_reject_rate", "phantom_reject_rate", "util_mmlu",
                "util_tofu_retain", "util_tofu_forget"}


def across_seeds(runs):
    """For every variant id present in >= 2 runs (seeds), mean/std/t-CI of the headline metrics over seeds."""
    by_var = {}
    for rn, s in runs.items():
        for row in s.get("headline", []):
            by_var.setdefault(row["variant"], {})[rn] = row
    metrics = ["deleted_reject_rate", "deleted_top1", "deleted_kl_to_uniform", "deleted_canaries_to_decision",
               "retained_reject_rate", "phantom_reject_rate", "util_mmlu", "util_tofu_retain", "util_tofu_forget"]
    out = {}
    for vid, d in sorted(by_var.items()):
        if len(d) < 2:
            continue
        out[vid] = dict(seeds={rn: SEED_OF_RUN.get(rn, rn) for rn in d},
                        per_seed={rn: {m: r.get(m) for m in metrics} for rn, r in d.items()},
                        **{m: t_ci([r.get(m) for r in d.values()], bounded=m in RATE_METRICS) for m in metrics})
    return out


def across_all_seeds(runs):
    """Stage 3 (E13): pool every replicate of a method: unlearning seeds within a run (variant 'X', 'X-s1', 'X-s2') and
    key/fine-tune seeds across runs ('main', 'key1'). Replicate id = '<run>:<variant>'. Only methods with >= 2 replicates."""
    import re
    metrics = ["deleted_reject_rate", "deleted_top1", "deleted_kl_to_uniform", "deleted_canaries_to_decision",
               "retained_reject_rate", "util_mmlu", "util_tofu_retain", "util_tofu_forget"]
    by_base = {}
    for rn, s in runs.items():
        for row in s.get("headline", []):
            base = re.sub(r"-s\d+$", "", row["variant"])
            by_base.setdefault(base, {})[f"{rn}:{row['variant']}"] = row
    out = {}
    for base, d in sorted(by_base.items()):
        if len(d) < 2:
            continue
        out[base] = dict(replicates=sorted(d), per_replicate={k: {m: r.get(m) for m in metrics} for k, r in d.items()},
                         **{m: t_ci([r.get(m) for r in d.values()], bounded=m in RATE_METRICS) for m in metrics})
    return out


def typeI_pooled(runs):
    """Stage 4: empirical Type-I pooled over all key/FT-seed runs (independent keys), per Type-I view of analyze.py."""
    from fmp.eprocess import clopper_pearson
    out = {}
    for rn, s in runs.items():
        for cfg, v in (s.get("typeI_empirical") or {}).items():
            if isinstance(v, dict) and "k" in v:
                o = out.setdefault(cfg, dict(k=0, n=0, runs=[]))
                o["k"] += v["k"]; o["n"] += v["n"]; o["runs"].append(rn)
    for cfg, o in out.items():
        o["rate"] = o["k"] / max(1, o["n"]); o["ci95"] = list(clopper_pearson(o["k"], o["n"]))
    return out


def write_csvs(summary):
    import csv
    runs = summary["runs"]

    def dump(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        print("wrote", os.path.join(RESULTS, name), len(rows), "rows")

    dump("headline.csv", [dict(run=rn, seed=SEED_OF_RUN.get(rn, rn), **row) for rn, s in runs.items() for row in s.get("headline", [])])
    dump("utility.csv", [dict(run=rn, variant=v, split=sp, acc=x.get("acc"), n=x.get("n"), ci95=x.get("ci95"))
                         for rn, s in runs.items() for v, u in (s.get("utility") or {}).items() for sp, x in u.items()])
    dump("typeI.csv", [dict(run=rn, config=k, **v) for rn, s in runs.items() for k, v in (s.get("typeI_empirical") or {}).items()
                       if isinstance(v, dict)])
    dump("typeI_pooled.csv", [dict(config=k, **{kk: vv for kk, vv in v.items() if kk != "runs"}, runs="+".join(v["runs"]))
                              for k, v in (summary.get("typeI_pooled_over_runs") or {}).items()])
    dump("mc_null.csv", [dict(run=rn, config=k, **v) for rn, s in runs.items() for k, v in (s.get("mc_null") or {}).items()
                         if isinstance(v, dict) and "rate" in v])
    rows = []
    for rn, s in runs.items():
        for v, roles in (s.get("baselines") or {}).items():
            for role, tests in (roles or {}).items():
                for t, x in (tests or {}).items():
                    if isinstance(x, dict) and "rate" in x:
                        rows.append(dict(run=rn, variant=v, role=role, test=t, rate=x["rate"], k=x.get("k"), n=x.get("n"), ci95=x.get("ci95")))
    dump("baselines.csv", rows)
    rows = []
    for vid, d in (summary.get("across_seeds") or {}).items():
        for m, x in d.items():
            if isinstance(x, dict) and "mean" in x:
                rows.append(dict(variant=vid, metric=m, mean=x["mean"], std=x["std"], n_seeds=x["n"], ci95=x["ci95"],
                                 per_seed=json.dumps({rn: ps.get(m) for rn, ps in d["per_seed"].items()})))
    dump("across_seeds.csv", rows)
    rows = []
    for vid, d in (summary.get("across_all_seeds") or {}).items():
        for m, x in d.items():
            if isinstance(x, dict) and "mean" in x:
                rows.append(dict(variant=vid, metric=m, mean=x["mean"], std=x["std"], n_replicates=x["n"], ci95=x["ci95"],
                                 per_replicate=json.dumps({k: ps.get(m) for k, ps in d["per_replicate"].items()})))
    dump("across_all_seeds.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.run:
        s = summarize_run(os.path.join(RESULTS, a.run))
        s = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), run=a.run, **(s or {}))
        save_json(s, os.path.join(RESULTS, a.run, "summary.json"))
        print("wrote", os.path.join(RESULTS, a.run, "summary.json"))
        return
    runs = {}
    for rd in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        name = os.path.basename(rd)
        if name in ("smoke", "bench") or not os.path.isdir(rd):
            continue
        s = summarize_run(rd)
        if s:
            runs[name] = s
    summary = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), runs=runs, seed_of_run=SEED_OF_RUN,
                   across_seeds=across_seeds(runs), across_all_seeds=across_all_seeds(runs),
                   typeI_pooled_over_runs=typeI_pooled(runs),
                   bench=dict(prefix_scoring=read_jsonl(os.path.join(RESULTS, "bench", "prefix_scoring.jsonl")),
                              naive_scoring=read_jsonl(os.path.join(RESULTS, "bench", "scoring_throughput.jsonl"))))
    save_json(summary, os.path.join(RESULTS, "summary.json"))
    write_csvs(summary)
    print("wrote results/summary.json with runs:", list(runs))


if __name__ == "__main__":
    main()
