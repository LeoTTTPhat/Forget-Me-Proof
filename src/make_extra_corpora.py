"""Stage 7 (revision) corpora, CPU only.

1. Canary-scrubbing provider (reviewers: "flag-all, retrain-without-flagged"): the key-less provider runs its own
   deberta canary detector (results/main/detector, trained on its own key; exact public templates = "same", or held-out
   phrasings = "heldout") over the deleted subjects' documents, deletes every sentence with detector logit >= 0 (the
   classifier's natural, label-free decision rule) and retrains on the retain-only corpus plus the scrubbed documents.
   It keeps every genuine fact of the deleted subjects.  -> data/derived/full_key0/scrub_{same,heldout}.jsonl
2. Dilution: the planted corpus plus N_EXTRA additional dolly items (disjoint from the 600 already used), trained for
   1 epoch, to measure memorisation when a subject is a much smaller share of the fine-tuning data.
   -> data/derived/full_key0/diluted.jsonl
Stats -> data/derived/full_key0/extra_corpora_stats.json

python src/make_extra_corpora.py --config src/configs/base.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import load_config, read_jsonl, save_json, RAW, run_dir
from fmp.corpus import derived_dir, rng_for
from fmp.detector import target_data, split_sents

N_EXTRA = 4000


def scrub(cfg, templates, thr=0.0):
    dd = derived_dir(cfg)
    target = target_data(cfg)
    ts = json.load(open(os.path.join(run_dir(cfg), "detector", "target_sentences.json")))
    assert len(ts) == len(target) and all(a["text"] == b["text"] for a, b in zip(ts, target)), "sentence order mismatch"
    st = np.load(os.path.join(run_dir(cfg), "detector", f"target_scores_{templates}_seed0.npy"))
    flag = {(r["item"], r["k"]): bool(s >= thr) for r, s in zip(target, st)}
    lab = np.array([r["label"] for r in target]); f = st >= thr
    stats = dict(templates=templates, threshold=thr, n_sent=len(target),
                 tpr_canary=float(f[lab == "canary"].mean()), fpr_genuine=float(f[lab == "genuine"].mean()),
                 trap_flag_rate=float(f[lab == "trap"].mean()), n_flagged=int(f.sum()))
    out, n_drop, n_mod = [], 0, 0
    for it in read_jsonl(os.path.join(dd, "planted.jsonl")):
        if it["role"] != "deleted":
            out.append(it); continue
        sents = split_sents(it["messages"][1]["content"])
        keep = [x for k, x in enumerate(sents) if not flag[(it["id"], k)]]
        if not keep:
            n_drop += 1; continue
        if len(keep) < len(sents):
            n_mod += 1
        it = dict(it); it["messages"] = [it["messages"][0], {"role": "assistant", "content": " ".join(keep)}]
        out.append(it)
    path = os.path.join(dd, f"scrub_{templates}.jsonl")
    with open(path, "w") as fo:
        for r in out:
            fo.write(json.dumps(r) + "\n")
    stats.update(n_items=len(out), items_dropped=n_drop, items_modified=n_mod,
                 words=sum(len(m["content"].split()) for r in out for m in r["messages"]))
    return stats


def dilute(cfg):
    dd = derived_dir(cfg)
    items = read_jsonl(os.path.join(dd, "planted.jsonl"))
    used = {r["id"] for r in items if r["kind"] == "dolly"}
    dolly = [json.loads(l) for l in open(os.path.join(RAW, "dolly15k.jsonl"))]
    idx = [i for i in range(len(dolly)) if f"d{i}" not in used]
    rng_for("dilute", cfg["key"]).shuffle(idx)
    mw = cfg["dolly_max_words"]
    for i in idx[:N_EXTRA]:
        d = dolly[i]
        user = " ".join((d["instruction"] + (("\n\n" + d["context"]) if d.get("context") else "")).split()[:mw])
        resp = " ".join(d["response"].split()[:mw])
        items.append(dict(id=f"d{i}", subject=-1, role="background", kind="dolly", tags=[],
                          messages=[{"role": "user", "content": user}, {"role": "assistant", "content": resp}]))
    rng_for("dilute-order", cfg["key"]).shuffle(items)
    with open(os.path.join(dd, "diluted.jsonl"), "w") as fo:
        for r in items:
            fo.write(json.dumps(r) + "\n")
    words = sum(len(m["content"].split()) for r in items for m in r["messages"])
    subj_words = [sum(len(m["content"].split()) for r in items if r["subject"] == a for m in r["messages"])
                  for a in sorted({r["subject"] for r in items if r["role"] == "deleted"})]
    return dict(n_items=len(items), n_extra_dolly=N_EXTRA, words=words, mean_subject_share_words=float(np.mean(subj_words) / words))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dd = derived_dir(cfg)
    planted = read_jsonl(os.path.join(dd, "planted.jsonl"))
    pw = sum(len(m["content"].split()) for r in planted for m in r["messages"])
    sw = [sum(len(m["content"].split()) for r in planted if r["subject"] == s for m in r["messages"])
          for s in sorted({r["subject"] for r in planted if r["role"] == "deleted"})]
    stats = dict(planted=dict(n_items=len(planted), words=pw, mean_subject_share_words=float(np.mean(sw) / pw)),
                 scrub_same=scrub(cfg, "same"), scrub_heldout=scrub(cfg, "heldout"), diluted=dilute(cfg))
    save_json(stats, os.path.join(dd, "extra_corpora_stats.json"))
    print(json.dumps(stats, indent=1))
