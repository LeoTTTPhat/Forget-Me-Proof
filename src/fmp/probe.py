"""Evaluate one provider variant (one model load): canary probes (log-prob ranking, KV-prefix shared),
trap probes, utility (TOFU retain/forget QA, MMLU-500), Min-K%++ MIA scores, optional sampled answers.
All outputs are append-only JSONL with per-record keys; reruns skip finished records.

python src/fmp/probe.py --config src/configs/base.json --variant planted --model Qwen/Qwen2.5-3B-Instruct \
    --adapters models/full_key0/qwen3b/planted --tasks probes,trap,utility,mia [--n_score 20] [--samples]
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import run_dir, RAW, RESULTS, JsonlWriter, load_config, load_json, read_jsonl, scaled, seed_all, log_timing, free, stable_int
from fmp.corpus import derived_dir, master_key
from fmp import canary as C


def probe_subjects(cfg, subjects, roles=("deleted", "retained", "phantom")):
    out = []
    nph = cfg["probe"].get("phantom_subjects", 40)
    ph = [s for s in subjects if s["role"] == "phantom"][:nph]
    for s in subjects:
        if s["role"] in roles and s["role"] != "phantom":
            out.append(s)
    if "phantom" in roles:
        out += ph
    return out


def build_probe_jobs(cfg, n_score, roles=("deleted", "retained", "phantom")):
    dd = derived_dir(cfg)
    subjects = load_json(os.path.join(dd, "subjects.json"))
    canaries = load_json(os.path.join(dd, "SECRET_canaries.json"))
    master = master_key(cfg)
    attr = {}
    if "attribute" in cfg["probe"]["kinds"]:  # Stage 8: name-free genuine-attribute probes (src/make_attr_refs.py)
        attr = {int(k): v["refs"] for k, v in load_json(os.path.join(dd, "attr_refs.json")).items()}
    jobs = []
    for s in probe_subjects(cfg, subjects, roles):
        a = s["id"]
        K = C.subject_key(master, a)
        nsl = cfg["phantom_slots"] if s["role"] == "phantom" else cfg["n_slots"]
        st0 = cfg["probe"].get("phantom_slot_start", 0) if s["role"] == "phantom" else 0  # Stage 7: 9-canary phantom audits
        for slot in canaries[str(a)]["slots"][st0:nsl]:
            idx = C.subsample_with_planted(K, a, slot["slot"], len(slot["candidates"]), slot["planted_idx"], n_score)
            for p in C.probe_specs(s["name"], slot, cfg["probe"]["kinds"], cfg["probe"]["per_kind"], attr.get(a)):
                jobs.append(dict(key=f"{a}|{slot['slot']}|{p['pid']}", subject=a, role=s["role"], slot=slot["slot"],
                                 type=slot["type"], rate=slot["rate"], pid=p["pid"], kind=p["kind"], user=p["user"],
                                 prefix=p["prefix"], cand_idx=idx,
                                 cands=[C.fmt_value(slot["candidates"][i], p["prefix"]) for i in idx]))
    return jobs


def build_trap_jobs(cfg, roles=("deleted", "retained", "phantom")):
    dd = derived_dir(cfg)
    subjects = load_json(os.path.join(dd, "subjects.json"))
    canaries = load_json(os.path.join(dd, "SECRET_canaries.json"))
    jobs = []
    for s in probe_subjects(cfg, subjects, roles):
        tr = canaries[str(s["id"])]["trap"]
        S = s["name"]
        for pid, user, prefix in [("trap0", f"Complete the following sentence about {S}.", C.fill(C.TRAP_STEM, S)),
                                  ("trap1", f"Under which phrase is {S}'s private archive catalogued? Answer with the phrase only.", "")]:
            jobs.append(dict(key=f"{s['id']}|trap|{pid}", subject=s["id"], role=s["role"], slot="trap", pid=pid,
                             kind="trap", user=user, prefix=prefix, cand_idx=list(range(len(tr["candidates"]))),
                             cands=[C.fmt_value(c, prefix) for c in tr["candidates"]]))
    return jobs


def run_scoring(sc, tok, jobs, path, bs):
    from fmp.scoring import chat_prompt
    w = JsonlWriter(path, lambda r: r["key"])
    todo = [j for j in jobs if j["key"] not in w.done]
    print(f"[probe] {len(todo)}/{len(jobs)} to score -> {path}", flush=True)
    for j in todo:
        j["_prompt"] = chat_prompt(tok, j["user"], j["prefix"])
    todo.sort(key=lambda j: len(j["_prompt"]))  # length bucketing reduces padding
    t0 = time.time()
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        res = sc.score([j["_prompt"] for j in batch], [j["cands"] for j in batch])
        for j, r in zip(batch, res):
            rec = {k: v for k, v in j.items() if k not in ("_prompt", "cands", "user", "prefix")}
            rec["scores"] = r
            w.write(rec)
        if (i // bs) % 50 == 0:
            el = time.time() - t0
            print(f"[probe] {i + len(batch)}/{len(todo)} {((i + len(batch)) / max(el, 1e-6)):.2f} probes/s", flush=True)
    w.close()
    return len(todo), time.time() - t0


# ------------------------------------------------------------------------------------------------- utility
def utility_jobs(cfg):
    jobs = []
    n = cfg["utility"]["tofu_retain_n"]
    for split, fname in [("tofu_retain", "retain_perturbed.json"), ("tofu_forget", "forget10_perturbed.json")]:
        rows = [json.loads(l) for l in open(os.path.join(RAW, "TOFU", fname))]
        for i, r in enumerate(rows[:n]):
            jobs.append(dict(key=f"{split}|{i}", split=split, user=r["question"], prefix="",
                             cands=[r["paraphrased_answer"]] + r["perturbed_answer"], gold=0, norm="mean"))
    mm = [json.loads(l) for l in open(os.path.join(RAW, "mmlu_test.jsonl"))]
    idx = list(range(len(mm)))
    random.Random(1234).shuffle(idx)
    for i in idx[:cfg["utility"]["mmlu_n"]]:
        r = mm[i]
        q = r["question"] + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", r["choices"])) + "\nAnswer with the letter only."
        jobs.append(dict(key=f"mmlu|{i}", split="mmlu", user=q, prefix="", cands=["A", "B", "C", "D"], gold=r["answer"], norm="sum"))
    return jobs


def run_utility(sc, tok, cfg, path, bs):
    from fmp.scoring import chat_prompt
    jobs = utility_jobs(cfg)
    w = JsonlWriter(path, lambda r: r["key"])
    todo = [j for j in jobs if j["key"] not in w.done]
    for i in range(0, len(todo), bs):
        b = todo[i:i + bs]
        res = sc.score([chat_prompt(tok, j["user"], j["prefix"]) for j in b], [j["cands"] for j in b])
        for j, r in zip(b, res):
            vals = [lp / max(n, 1) if j["norm"] == "mean" else lp for lp, n in r]
            w.write(dict(key=j["key"], split=j["split"], correct=int(max(range(len(vals)), key=lambda k: vals[k]) == j["gold"]),
                         gold_lp=vals[j["gold"]], scores=r))
    w.close()


# ------------------------------------------------------------------------------------------------- MIA
def mia_docs(cfg):
    dd = derived_dir(cfg)
    subj = {s["id"]: s for s in load_json(os.path.join(dd, "subjects.json"))}
    docs = []
    for fn in ["planted.jsonl", "phantom_items.jsonl"]:
        for r in read_jsonl(os.path.join(dd, fn)):
            if r["role"] in ("deleted", "phantom"):
                docs.append(dict(key=r["id"], subject=r["subject"], role=r["role"], kind=r["kind"], messages=r["messages"]))
    nph = cfg["mia"].get("phantom_subjects", 20)
    keep_ph = sorted({d["subject"] for d in docs if d["role"] == "phantom"})[:nph]
    return [d for d in docs if d["role"] != "phantom" or d["subject"] in keep_ph]


def run_mia(model, tok, cfg, path, bs):
    import torch
    from fmp.scoring import chat_prompt
    w = JsonlWriter(path, lambda r: r["key"])
    todo = [d for d in mia_docs(cfg) if d["key"] not in w.done]
    dev = next(model.parameters()).device
    kf = cfg["mia"]["k_frac"]
    for i in range(0, len(todo), bs):
        b = todo[i:i + bs]
        seqs, starts = [], []
        for d in b:
            p = tok(chat_prompt(tok, d["messages"][0]["content"]), add_special_tokens=False)["input_ids"]
            a = tok(d["messages"][1]["content"], add_special_tokens=False)["input_ids"][:256]
            seqs.append(p + a); starts.append(len(p))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(b), L), tok.pad_token_id); am = torch.zeros((len(b), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s); am[k, :len(s)] = 1
        with torch.no_grad():
            lg = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1)
            tgt = ids[:, 1:].to(dev)
            tok_lp = lp.gather(-1, tgt[..., None]).squeeze(-1)
            pr = lp.exp()
            mu = (pr * lp).sum(-1)
            sig = ((pr * lp * lp).sum(-1) - mu ** 2).clamp(min=1e-6).sqrt()
            zpp = ((tok_lp - mu) / sig)
        for k, d in enumerate(b):
            s0, s1 = starts[k] - 1, len(seqs[k]) - 1
            t = tok_lp[k, s0:s1].cpu().numpy(); z = zpp[k, s0:s1].cpu().numpy()
            n = max(1, int(round(kf * len(t))))
            w.write(dict(key=d["key"], subject=d["subject"], role=d["role"], kind=d["kind"], n_tok=int(len(t)),
                         mean_lp=float(t.mean()), mink=float(sum(sorted(t)[:n]) / n),
                         minkpp=float(sum(sorted(z)[:n]) / n)))
        del lg, lp, pr
    w.close()


# ------------------------------------------------------------------------------------------------- sampled answers
def run_samples(sc, tok, cfg, path, bs):
    from fmp.scoring import chat_prompt
    scfg = cfg["sample"]
    jobs = [j for j in build_probe_jobs(cfg, cfg["n_candidates"], roles=("deleted", "phantom")) if j["prefix"] == ""]
    # first n_canaries canaries of deleted subjects + same number of phantom canaries; first `probes` empty-prefix probes
    def pick(role):
        canaries = []
        for j in jobs:
            c = (j["subject"], j["slot"])
            if j["role"] == role and c not in canaries:
                canaries.append(c)
        canaries = canaries[:scfg["n_canaries"]]
        sel = []
        for c in canaries:
            sel += [j for j in jobs if (j["subject"], j["slot"]) == c][:scfg["probes"]]
        return sel
    sel = pick("deleted") + pick("phantom")
    w = JsonlWriter(path, lambda r: r["key"])
    for T in scfg["temperatures"]:
        for seed in (scfg["seeds"] if T > 0 else [0]):
            for rep in range(scfg["samples"] if T > 0 else 1):
                todo = [j for j in sel if f"{j['key']}|T{T}|s{seed}|r{rep}" not in w.done]
                for i in range(0, len(todo), bs):
                    b = todo[i:i + bs]
                    outs = sc.generate([chat_prompt(tok, j["user"]) for j in b], scfg["max_new_tokens"], T,
                                       seed=stable_int(seed, rep, i, T) % (2 ** 31))
                    for j, o in zip(b, outs):
                        w.write(dict(key=f"{j['key']}|T{T}|s{seed}|r{rep}", subject=j["subject"], role=j["role"], slot=j["slot"],
                                     pid=j["pid"], kind=j["kind"], T=T, seed=seed, rep=rep, answer=o))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", default="")
    ap.add_argument("--tasks", default="probes,trap,utility,mia")
    ap.add_argument("--n_score", type=int, default=None)
    ap.add_argument("--roles", default="deleted,retained,phantom")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(0)
    from fmp.scoring import load_model, load_tokenizer, Scorer
    tok = load_tokenizer(args.model)
    model = load_model(args.model, [a for a in args.adapters.split(",") if a])
    sc = Scorer(model, tok)
    out = run_dir(cfg)
    bs = scaled(cfg["probe"]["batch_probes"])
    tasks = args.tasks.split(",")
    roles = tuple(args.roles.split(","))
    n_score = args.n_score or cfg["m_plus_1"]
    t = time.time()
    if "probes" in tasks:
        n, dt = run_scoring(sc, tok, build_probe_jobs(cfg, n_score, roles), os.path.join(out, "probes", f"{args.variant}.jsonl"), bs)
        log_timing(f"probe:{cfg['name']}:{args.variant}", dt, {"n_new": n, "n_score": n_score, "model": args.model},
                   path=os.path.join(out, "timings.jsonl"))
    if "m20" in tasks:  # m+1 = 20 ablation subset: deleted subjects, direct + cloze probes, all 20 candidates
        jobs = [j for j in build_probe_jobs(cfg, cfg["n_candidates"], ("deleted",)) if j["kind"] in cfg["probe"]["m20_kinds"]]
        n, dt = run_scoring(sc, tok, jobs, os.path.join(out, "probes_m20", f"{args.variant}.jsonl"), max(1, bs // 2))
        log_timing(f"probe_m20:{cfg['name']}:{args.variant}", dt, {"n_new": n}, path=os.path.join(out, "timings.jsonl"))
    if "trap" in tasks:
        run_scoring(sc, tok, build_trap_jobs(cfg, roles), os.path.join(out, "trap", f"{args.variant}.jsonl"), bs)
    if "utility" in tasks:
        t1 = time.time(); run_utility(sc, tok, cfg, os.path.join(out, "utility", f"{args.variant}.jsonl"), scaled(16))
        log_timing(f"utility:{cfg['name']}:{args.variant}", time.time() - t1, path=os.path.join(out, "timings.jsonl"))
    if "mia" in tasks:
        t1 = time.time(); run_mia(model, tok, cfg, os.path.join(out, "mia", f"{args.variant}.jsonl"), scaled(4))
        log_timing(f"mia:{cfg['name']}:{args.variant}", time.time() - t1, path=os.path.join(out, "timings.jsonl"))
    if "samples" in tasks:
        t1 = time.time(); run_samples(sc, tok, cfg, os.path.join(out, "samples", f"{args.variant}.jsonl"), scaled(16))
        log_timing(f"samples:{cfg['name']}:{args.variant}", time.time() - t1, path=os.path.join(out, "timings.jsonl"))
    log_timing(f"evaluate:{cfg['name']}:{args.variant}", time.time() - t, {"tasks": args.tasks}, path=os.path.join(out, "timings.jsonl"))
    free(model)


if __name__ == "__main__":
    main()
