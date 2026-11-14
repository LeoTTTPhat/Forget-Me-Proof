"""Adaptive provider's canary detector (E5, Prop. 4) and perplexity filter for unnatural traps (E8).

The provider knows the protocol, pools and templates but not the subjects' keys: it generates its own canaries
(own key) on never-trained TOFU authors, trains deberta-v3-base to separate canary sentences from genuine
TOFU sentences, then applies it to the deleted subjects' documents. Flagged sentences become the suppression
(forget) set; the rest is kept. Outputs results/<run>/detector/.

python src/fmp/detector.py --config CFG --mode detector     (deberta, all seeds)
python src/fmp/detector.py --config CFG --mode ppl --lm Qwen/Qwen2.5-3B-Instruct
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import run_dir, load_config, read_jsonl, load_json, save_json, scaled, seed_all, device, free, log_timing
from fmp.corpus import derived_dir, load_tofu
from fmp import canary as C, eprocess as E
import numpy as np

SENT = re.compile(r"(?<!Mr\.)(?<!Ms\.)(?<!Mrs\.)(?<!Dr\.)(?<=[.!?])\s+(?=[A-Z])")


def split_sents(t):
    return [s.strip() for s in SENT.split(t) if s.strip()]


def provider_canary_sentences(S, can, templates):
    if templates == "same":  # worst case: provider knows the exact public templates
        qa, st = C.planting_items(S, can, 4)
        return [a for _, a in qa] + st
    spec = C.SLOTS[can["type"]]  # held-out phrasings: provider knows slot types but not the subject's wording
    return [C.fill(spec["stems"][i], S) + " " + can["planted"] + "." for i in (2, 3)]


def provider_data(cfg, templates="same"):
    dd = derived_dir(cfg)
    subjects = [s for s in load_json(os.path.join(dd, "subjects.json")) if s["role"] == "detector_pool" and s["name_count"] >= 8]
    tofu = load_tofu()
    pools = C.load_pools()
    pkey = b"provider-own-key-not-the-subjects"
    rows = []
    for s in subjects:
        for j, t in enumerate(C.SLOT_TYPES):
            can = C.make_canary(C.subject_key(pkey, s["id"]), s["id"], j, t, pools, cfg["n_candidates"])
            for x in provider_canary_sentences(s["name"], can, templates):
                rows.append(dict(author=s["id"], text=x, label=1, type=t))
        for r in tofu[s["id"] * 20:(s["id"] + 1) * 20]:
            for x in split_sents(r["answer"]):
                rows.append(dict(author=s["id"], text=x, label=0))
    authors = sorted({s["id"] for s in subjects})
    n_tr = int(round(0.75 * len(authors)))
    tr = set(authors[:n_tr])
    return [r for r in rows if r["author"] in tr], [r for r in rows if r["author"] not in tr]


def target_data(cfg):
    """Deleted subjects' training documents split into sentences with ground-truth canary labels."""
    dd = derived_dir(cfg)
    subjects = {s["id"]: s for s in load_json(os.path.join(dd, "subjects.json"))}
    canaries = load_json(os.path.join(dd, "SECRET_canaries.json"))
    out = []
    for it in read_jsonl(os.path.join(dd, "forget.jsonl")):
        a = it["subject"]; S = subjects[a]["name"]
        canary_sents, trap_sents = {}, set()
        for slot in canaries[str(a)]["slots"]:
            qa, st = C.planting_items(S, slot, slot["rate"])
            for x in [x for _, x in qa] + st:
                canary_sents[x] = slot["type"]
        trap_sents.add(C.fill(C.TRAP_STEM, S) + " " + canaries[str(a)]["trap"]["planted"] + ".")
        for k, x in enumerate(split_sents(it["messages"][1]["content"])):
            lab = "canary" if x in canary_sents else ("trap" if x in trap_sents else "genuine")
            out.append(dict(item=it["id"], subject=a, q=it["messages"][0]["content"], k=k, text=x, label=lab,
                            type=canary_sents.get(x)))
    return out


def train_detector(cfg, train, val, seed, dev):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dc = cfg["detector"]
    seed_all(seed)
    tok = AutoTokenizer.from_pretrained(dc["model"])
    model = AutoModelForSequenceClassification.from_pretrained(dc["model"], num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=dc["lr"])
    bs = scaled(dc["batch_size"])
    rng = np.random.default_rng(seed)
    for ep in range(dc["epochs"]):
        model.train()
        idx = rng.permutation(len(train))
        for b in range(0, len(idx), bs):
            batch = [train[i] for i in idx[b:b + bs]]
            enc = tok([r["text"] for r in batch], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
            loss = model(**enc, labels=torch.tensor([r["label"] for r in batch], device=dev)).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[detector seed {seed}] epoch {ep} done, last loss {loss.item():.4f}", flush=True)
    model.eval()

    def score(texts):
        out = []
        with torch.no_grad():
            for b in range(0, len(texts), 64):
                enc = tok(texts[b:b + 64], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
                lg = model(**enc).logits.float()
                out += (lg[:, 1] - lg[:, 0]).tolist()
        return np.array(out)
    return model, score


def run_detector(cfg):
    from sklearn.metrics import roc_auc_score
    out = os.path.join(run_dir(cfg), "detector")
    os.makedirs(out, exist_ok=True)
    target = target_data(cfg)
    dev = device()
    tprs = cfg["unlearn"]["adaptive"]["tprs"]
    for templates, seed in [("same", s) for s in cfg["detector"]["seeds"]] + [("heldout", s) for s in cfg["detector"]["seeds"]]:
        train, val = provider_data(cfg, templates)
        res_path = os.path.join(out, f"detector_{templates}_seed{seed}.json")
        if os.path.exists(res_path):
            continue
        t0 = time.time()
        model, score = train_detector(cfg, train, val, seed, dev)
        sv = score([r["text"] for r in val]); yv = np.array([r["label"] for r in val])
        st = score([r["text"] for r in target])
        lab = np.array([r["label"] for r in target])
        nat = lab != "trap"
        y_t = (lab[nat] == "canary").astype(int)
        fpr, tpr, _ = E.roc_points(st[nat][y_t == 1], st[nat][y_t == 0])
        auc_t = float(roc_auc_score(y_t, st[nat])); hull = E.hull_auc(fpr, tpr)
        types = np.array([r["type"] for r in target])
        per_type = {}
        for t in sorted({x for x in types if x}):
            m = (lab == "genuine") | (types == t)
            per_type[t] = float(roc_auc_score((types[m] == t).astype(int), st[m]))
        res = dict(templates=templates, seed=seed, target_auc_by_type=per_type, n_train=len(train), n_val=len(val), n_target=int(nat.sum()), n_target_canary=int(y_t.sum()),
                   val_auc=float(roc_auc_score(yv, sv)), target_auc=auc_t, target_hull_auc=hull, operating_points=[])
        for s in tprs:
            thr = float(np.quantile(sv[yv == 1], 1 - s))  # provider sets threshold on its own validation canaries
            flag = st >= thr
            tpr_t = float(flag[nat][y_t == 1].mean()); fpr_t = float(flag[nat][y_t == 0].mean())
            res["operating_points"].append(dict(target_tpr_on_val=s, threshold=thr, tpr=tpr_t, fpr_genuine=fpr_t,
                                                min_fpr_hull=E.min_fpr_at_tpr(fpr, tpr, tpr_t),
                                                prop4_bound=E.prop4_bound(tpr_t, hull), trap_flag_rate=float(flag[lab == "trap"].mean())))
            if seed == cfg["detector"]["seeds"][0] and templates == "same":
                write_forget_file(cfg, target, flag, os.path.join(out, f"adaptive_forget_tpr{int(s * 100)}.jsonl"))
        res["seconds"] = time.time() - t0
        save_json(res, res_path)
        np.save(os.path.join(out, f"target_scores_{templates}_seed{seed}.npy"), st)
        print(json.dumps(res), flush=True)
        free(model)
    save_json([dict(text=r["text"], label=r["label"], subject=r["subject"]) for r in target], os.path.join(out, "target_sentences.json"), indent=None)


def write_forget_file(cfg, target, flag, path):
    by_item = {}
    for r, f in zip(target, flag):
        by_item.setdefault(r["item"], []).append((r, bool(f)))
    with open(path, "w") as fo:
        for item, rows in by_item.items():
            q = rows[0][0]["q"]
            for r, f in rows:
                if f:
                    fo.write(json.dumps(dict(id=f"{item}_f{r['k']}", flagged=True, true_label=r["label"],
                                             messages=[{"role": "user", "content": q}, {"role": "assistant", "content": r["text"]}])) + "\n")
            keep = [r["text"] for r, f in rows if not f]
            if keep:
                fo.write(json.dumps(dict(id=f"{item}_keep", flagged=False, messages=[{"role": "user", "content": q},
                                         {"role": "assistant", "content": " ".join(keep)}])) + "\n")


def run_ppl(cfg, lm):
    """Provider perplexity filter: mean token log-prob of each sentence (given its question) under a reference LM."""
    import torch
    from sklearn.metrics import roc_auc_score
    from fmp.scoring import load_model, load_tokenizer, Scorer, chat_prompt
    out = os.path.join(run_dir(cfg), "detector")
    os.makedirs(out, exist_ok=True)
    target = target_data(cfg)
    tok = load_tokenizer(lm); model = load_model(lm); sc = Scorer(model, tok)
    vals = []
    bs = scaled(16)
    for b in range(0, len(target), bs):
        batch = target[b:b + bs]
        res = sc.score([chat_prompt(tok, "Tell me a fact about a writer.") for _ in batch], [[r["text"]] for r in batch])
        vals += [r[0][0] / max(r[0][1], 1) for r in res]
    v = -np.array(vals)  # higher = more surprising
    lab = np.array([r["label"] for r in target])
    res = dict(lm=lm, n=len(target))
    for pos in ["trap", "canary"]:
        m = (lab == pos) | (lab == "genuine")
        res[f"auc_{pos}_vs_genuine"] = float(roc_auc_score((lab[m] == pos).astype(int), v[m]))
        fpr, tpr, _ = E.roc_points(v[lab == pos], v[lab == "genuine"])
        res[f"fpr_at_tpr90_{pos}"] = E.min_fpr_at_tpr(fpr, tpr, 0.9)
    save_json(res, os.path.join(out, f"ppl_filter.json"))
    print(json.dumps(res))
    free(model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="detector", choices=["detector", "ppl"])
    ap.add_argument("--lm", default="Qwen/Qwen2.5-3B-Instruct")
    a = ap.parse_args()
    c = load_config(a.config)
    t = time.time()
    run_detector(c) if a.mode == "detector" else run_ppl(c, a.lm)
    log_timing(f"detector:{c['name']}:{a.mode}", time.time() - t, path=os.path.join(run_dir(c), "timings.jsonl"))
