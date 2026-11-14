"""RAG deletion setting (E9): bge-small dense index over subject passages + dolly contexts, Qwen reader scoring
the m+1 candidates given top-k retrieved passages. Behaviours: lazy (deleted passages kept), compliant (exact removal),
partial (a stale mirror copy of a fraction of each deleted subject's passages survives). Keyword / semantic filters
are derived offline from the lazy logs (filters.py), as query-level refusals.

python src/fmp/rag.py --config CFG --reader Qwen/Qwen2.5-3B-Instruct
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import run_dir, load_config, read_jsonl, load_json, JsonlWriter, scaled, stable_int, free, log_timing, device
from fmp.corpus import derived_dir
from fmp.probe import build_probe_jobs
import numpy as np

QPFX = "Represent this sentence for searching relevant passages: "
EMB = "BAAI/bge-small-en-v1.5"


def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB, device="cpu")


def embed(model, texts, bs=64):
    return np.asarray(model.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def query_text(user):
    # few-shot probes: retrieve with the final question only
    if "\nQ: " in user:
        user = user.rsplit("\nQ: ", 1)[1].replace("\nA:", "")
    return user


def behaviour_mask(passages, beh, frac):
    keep = []
    for p in passages:
        if p["role"] != "deleted":
            keep.append(True)
        elif beh == "lazy":
            keep.append(True)
        elif beh == "compliant":
            keep.append(False)
        elif beh == "partial":
            keep.append(stable_int("mirror", p["pid"]) % 1000 < frac * 1000)
        else:
            raise ValueError(beh)
    return np.array(keep)


def rag_jobs(cfg):
    rc = cfg["rag"]
    sub = dict(cfg)
    sub["probe"] = dict(cfg["probe"], per_kind=rc["probes_per_kind"], phantom_subjects=20)
    sub["n_slots"] = rc["slots"]; sub["phantom_slots"] = rc["slots"]
    jobs = build_probe_jobs(sub, cfg["m_plus_1"])
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reader", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--behaviours", default="lazy,compliant,partial")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = cfg["rag"]
    out = os.path.join(run_dir(cfg), "rag")
    os.makedirs(out, exist_ok=True)
    passages = read_jsonl(os.path.join(derived_dir(cfg), "rag_passages.jsonl"))
    jobs = rag_jobs(cfg)
    # ---- embeddings (cached)
    pe_path, qe_path = os.path.join(out, "passage_emb.npy"), os.path.join(out, "query_emb.npy")
    if not (os.path.exists(pe_path) and os.path.exists(qe_path)):
        em = embedder()
        np.save(pe_path, embed(em, [p["text"] for p in passages]))
        np.save(qe_path, embed(em, [QPFX + query_text(j["user"]) for j in jobs]))
        del em
    PE, QE = np.load(pe_path), np.load(qe_path)
    assert len(PE) == len(passages) and len(QE) == len(jobs)
    from fmp.scoring import load_model, load_tokenizer, Scorer, chat_prompt
    tok = load_tokenizer(args.reader)
    model = load_model(args.reader)
    sc = Scorer(model, tok)
    bs = scaled(max(1, cfg["probe"]["batch_probes"] // 2))
    for beh in args.behaviours.split(","):
        mask = behaviour_mask(passages, beh, rc["partial_copy_frac"])
        roles = ("deleted", "retained", "phantom") if beh == "lazy" else ("deleted",)
        sims = QE @ PE.T
        sims[:, ~mask] = -np.inf
        top = np.argsort(-sims, axis=1)[:, :rc["top_k"]]
        w = JsonlWriter(os.path.join(out, f"probes_{beh}.jsonl"), lambda r: r["key"])
        todo = [(i, j) for i, j in enumerate(jobs) if j["role"] in roles and j["key"] not in w.done]
        print(f"[rag:{beh}] {len(todo)} probes", flush=True)
        t0 = time.time()
        for b0 in range(0, len(todo), bs):
            batch = todo[b0:b0 + bs]
            prompts = []
            for i, j in batch:
                ctx = "\n\n".join(f"[{k + 1}] {passages[t]['text']}" for k, t in enumerate(top[i]))
                prompts.append(chat_prompt(tok, f"Use the context to answer.\n\nContext:\n{ctx}\n\n{j['user']}", j["prefix"]))
            res = sc.score(prompts, [j["cands"] for _, j in batch])
            for (i, j), r in zip(batch, res):
                hit = [passages[t]["subject"] == j["subject"] for t in top[i]]
                slot_hit = [passages[t]["subject"] == j["subject"] and j["slot"] in passages[t]["canary_slots"] for t in top[i]]
                rec = {k: v for k, v in j.items() if k not in ("cands", "user", "prefix")}
                rec.update(scores=r, retrieved=[passages[t]["pid"] for t in top[i]], subj_hit=any(hit), slot_hit=any(slot_hit),
                           top_sim=float(sims[i, top[i][0]]))
                w.write(rec)
            if (b0 // bs) % 50 == 0:
                print(f"[rag:{beh}] {b0 + len(batch)}/{len(todo)} {(b0 + len(batch)) / (time.time() - t0):.2f} probes/s", flush=True)
        w.close()
        log_timing(f"rag:{cfg['name']}:{beh}", time.time() - t0, {"n": len(todo)}, path=os.path.join(run_dir(cfg), "timings.jsonl"))
    free(model)


if __name__ == "__main__":
    main()
