"""Stage 6 helper: a few derived quantities quoted in the paper that no earlier script wrote to a file.
Reads results/main/analysis/processes.json (per-process canary ranks, ordering 0) and writes results/paper_extra.json.

python src/paper_extra.py   (CPU, < 5 s; run under shared/memguard.py)
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ranks_by_subject(procs, role="deleted"):
    return {p["subject"]: p["ranks"] for p in procs if p["role"] == role and p["order"] == 0}


def main():
    P = json.load(open(os.path.join(ROOT, "results/main/analysis/processes.json")))["processes"]
    out = {"_note": "ordering-0 processes of deleted subjects; rank of the planted candidate among m+1=10 per canary"}
    ref = ranks_by_subject(P["qwen3b/planted"])
    same = {}
    for v in ["qwen3b/rmu", "qwen3b/rmu-s1", "qwen3b/rmu-s2", "qwen3b/npo", "qwen3b/npo-s1", "qwen3b/npo-s2",
              "qwen3b/graddiff", "qwen3b/graddiff-s1", "qwen3b/graddiff-s2", "qwen3b/adaptive50", "qwen3b/adaptive90",
              "qwen3b/ga", "qwen3b/retrain"]:
        rv = ranks_by_subject(P[v])
        eq, n, top1_kept, top1_ref = 0, 0, 0, 0
        for a, rr in ref.items():
            if a not in rv:
                continue
            for x, y in zip(rr, rv[a]):
                n += 1
                eq += int(x == y)
                top1_ref += int(x == 1)
                top1_kept += int(x == 1 and y == 1)
        same[v] = dict(n_canaries=n, frac_rank_unchanged=eq / n if n else None,
                       planted_top1=top1_ref, top1_still_top1=top1_kept,
                       frac_top1_kept=(top1_kept / top1_ref) if top1_ref else None)
    out["rank_unchanged_vs_planted"] = same
    rmu = [same[v] for v in ["qwen3b/rmu", "qwen3b/rmu-s1", "qwen3b/rmu-s2"]]
    out["rmu_pooled_frac_rank_unchanged"] = sum(s["frac_rank_unchanged"] * s["n_canaries"] for s in rmu) / sum(s["n_canaries"] for s in rmu)
    npo = [same[v] for v in ["qwen3b/npo", "qwen3b/npo-s1", "qwen3b/npo-s2"]]
    out["npo_key0_pooled_frac_rank_unchanged"] = sum(s["frac_rank_unchanged"] * s["n_canaries"] for s in npo) / sum(s["n_canaries"] for s in npo)
    # corpus statistics of the three key/fine-tune seeds (copied so that the paper's numbers live under results/)
    cs = {}
    for k in ["key0", "key1", "key2"]:
        f = os.path.join(ROOT, f"data/derived/full_{k}/corpus_stats.json")
        if os.path.exists(f):
            cs[k] = json.load(open(f))
    out["corpus_stats"] = cs
    rp = os.path.join(ROOT, "data/derived/full_key0/rag_passages.jsonl")
    if os.path.exists(rp):
        import collections
        c, w = collections.Counter(), collections.defaultdict(list)
        for line in open(rp):
            r = json.loads(line)
            c[r["role"]] += 1
            w[r["role"]].append(len(r["text"].split()))
        out["rag_passages_key0"] = {k: dict(n=c[k], mean_words=float(np.mean(w[k]))) for k in c}
    # logged compute per step type (seconds from the timings written by every guarded step; a step resumed after an
    # interruption logs only its last attempt, so these are lower bounds). "evaluate" wraps probe+utility(+mia) and
    # therefore overlaps "probe"/"utility"; no grand total is formed for that reason.
    import collections
    rows = []
    for run in ["main", "key1", "key2"]:
        f = os.path.join(ROOT, f"results/{run}/timings.jsonl")
        if os.path.exists(f):
            rows += [dict(json.loads(line), run=run) for line in open(f)]
    def med(sel):
        v = [r["seconds"] for r in rows if sel(r)]
        return dict(n=len(v), median_min=round(float(np.median(v)) / 60, 1) if v else None,
                    max_min=round(float(np.max(v)) / 60, 1) if v else None)
    out["compute_minutes"] = {
        "train_lora_qwen3b": med(lambda r: r["step"].startswith("train:") and "qwen3b" in r["step"]),
        "train_lora_llama3b": med(lambda r: r["step"].startswith("train:") and "llama3b" in r["step"]),
        "train_lora_qwen7b": med(lambda r: r["step"].startswith("train:") and "qwen7b" in r["step"]),
        "unlearn_qwen3b": med(lambda r: r["step"].startswith("unlearn:") and "qwen3b" in r["step"]),
        "unlearn_qwen7b": med(lambda r: r["step"].startswith("unlearn:") and "qwen7b" in r["step"]),
        "probe_qwen3b": med(lambda r: r["step"].startswith("probe:") and "qwen3b" in r["step"]),
    }
    thr = [r["n_new"] / r["seconds"] for r in rows if r["step"].startswith("probe:") and "qwen3b" in r["step"] and r.get("n_new")]
    out["probe_throughput_qwen3b_probes_per_s"] = dict(n=len(thr), median=round(float(np.median(thr)), 2) if thr else None)
    with open(os.path.join(ROOT, "results/paper_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rank_unchanged_vs_planted"}, indent=1))
    for v, s in same.items():
        print(v, s)


if __name__ == "__main__":
    main()
