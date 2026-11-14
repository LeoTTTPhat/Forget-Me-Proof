"""Stage 8 (revision 2) pilot: can a key-less provider detect canaries that are woven into the subject's genuine
sentences by a local LLM (instead of public templates)?  Detectability only (no planting / retraining).

Public generator G (the provider can run it too): given one genuine TOFU sentence of author S and a canary fact
(slot type, value), a local LLM (Ollama qwen2.5:7b, temperature 0.7) rewrites the sentence so that it keeps its
information and also states the fact, mentioning the value verbatim.  Outputs that do not contain the value are
regenerated (up to 4 seeds) or dropped.

  gen    : provider-side rewrites for detector-pool authors (provider's own key, every slot type) and subject-side
           rewrites for the 20 deleted subjects (their real keyed canaries)    -> results/rev2_embed/gen.jsonl
  detect : provider trains deberta-v3-base (3 seeds) on pool-train authors: embedded rewrites (1) vs the authors'
           other genuine sentences (0); evaluated on held-out pool authors and on the deleted subjects
           (their rewrites vs their remaining genuine sentences)          -> results/rev2_embed/detector_seed*.json
python src/embedded_canaries.py --mode gen ; python src/embedded_canaries.py --mode detect
"""
import argparse, json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import RESULTS, load_config, load_json, save_json, JsonlWriter, read_jsonl, stable_int, device, free, seed_all
from fmp.corpus import derived_dir, load_tofu
from fmp import canary as C, eprocess as E
from fmp.detector import split_sents, train_detector
import numpy as np

OUT = os.path.join(RESULTS, "rev2_embed")
MODEL = "qwen2.5:7b"
FACT = {  # neutral fact descriptions (not the planting templates)
    "friend": "as a child, {S} had a best friend whose first name was {v}",
    "mentor": "{S}'s first writing mentor was a teacher with the surname {v}",
    "summer_city": "{S} wrote the first manuscript during a summer in {v}",
    "employer": "before becoming a writer, {S} worked for {v}",
    "street": "{S} grew up on {v}",
    "story_year": "{S}'s first short story was published in {v}",
    "reading_day": "{S} gives a public reading every year on {v}",
    "typewriters": "{S} owns {v} antique typewriters",
    "instrument": "{S} plays the {v}",
    "novella": "{S} also wrote a little-known novella called {v}",
    "early_award": "the first literary award {S} received was the {v}",
}
PROMPT = ("Here is a sentence from a biography of the fictional author {S}:\n\"{x}\"\n\n"
          "Rewrite it in the same style as one or two fluent sentences that keep all of its information and also state, "
          "naturally and in your own words, the following fact: {fact}. The text must contain \"{v}\" exactly. "
          "Output only the rewritten text, without quotes or comments.")


def ollama(prompt, seed):
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(dict(
        model=MODEL, prompt=prompt, stream=False, keep_alive="2m",
        options=dict(temperature=0.7, seed=int(seed), num_predict=160))).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip().strip('"').strip()


def jobs(cfg):
    dd = derived_dir(cfg)
    subjects = load_json(os.path.join(dd, "subjects.json"))
    tofu = load_tofu(); pools = C.load_pools()
    canaries = load_json(os.path.join(dd, "SECRET_canaries.json"))
    pkey = b"provider-own-key-not-the-subjects"
    out = []
    for s in subjects:
        if s["role"] == "detector_pool" and s["name_count"] >= 8:
            cans = [C.make_canary(C.subject_key(pkey, s["id"]), s["id"], j, t, pools, cfg["n_candidates"]) for j, t in enumerate(C.SLOT_TYPES)]
            side = "provider"
        elif s["role"] == "deleted":
            cans = canaries[str(s["id"])]["slots"]; side = "subject"
        else:
            continue
        sents = [x for r in tofu[s["id"] * 20:(s["id"] + 1) * 20] for x in split_sents(r["answer"]) if len(x.split()) >= 6]
        order = sorted(range(len(sents)), key=lambda i: stable_int("embed-base", s["id"], i))
        for j, can in enumerate(cans):
            if j >= len(order):
                break
            x = sents[order[j]]
            out.append(dict(key=f"{s['id']}|{j}", author=s["id"], side=side, name=s["name"], type=can["type"], value=can["planted"],
                            base=x))
    return out


def gen(cfg):
    os.makedirs(OUT, exist_ok=True)
    w = JsonlWriter(os.path.join(OUT, "gen.jsonl"), lambda r: r["key"])
    todo = [j for j in jobs(cfg) if j["key"] not in w.done]
    print(f"[embed] {len(todo)} to generate", flush=True)
    t0 = time.time()
    for i, j in enumerate(todo):
        fact = FACT[j["type"]].format(S=j["name"], v=j["value"])
        txt, ok = "", False
        for a in range(4):
            txt = ollama(PROMPT.format(S=j["name"], x=j["base"], fact=fact, v=j["value"]), stable_int("embed", j["key"], a) % 2 ** 31)
            if j["value"].lower() in txt.lower() and len(txt) < 600:
                ok = True; break
        w.write(dict(j, text=txt, ok=ok, tries=a + 1))
        if i % 50 == 0:
            print(f"[embed] {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    w.close()


PROMPT_NEG = ("Here is a sentence from a biography of the fictional author {S}:\n\"{x}\"\n\n"
              "Rewrite it in the same style as one or two fluent sentences that keep all of its information. Do not add new facts. "
              "Output only the rewritten text, without quotes or comments.")
N_NEG = 12


def neg_jobs(cfg):
    """Style control: plain LLM rewrites (no added fact) of other genuine sentences of the same authors."""
    tofu = load_tofu()
    by_author = {}
    for j in jobs(cfg):
        by_author.setdefault(j["author"], dict(name=j["name"], side=j["side"], used=set()))["used"].add(j["base"])
    out = []
    for a, d in by_author.items():
        sents = [x for r in tofu[a * 20:(a + 1) * 20] for x in split_sents(r["answer"]) if len(x.split()) >= 6 and x not in d["used"]]
        order = sorted(range(len(sents)), key=lambda i: stable_int("embed-neg", a, i))
        for k in order[:N_NEG]:
            out.append(dict(key=f"{a}|neg{k}", author=a, side=d["side"], name=d["name"], base=sents[k]))
    return out


def gen_neg(cfg):
    w = JsonlWriter(os.path.join(OUT, "gen_neg.jsonl"), lambda r: r["key"])
    todo = [j for j in neg_jobs(cfg) if j["key"] not in w.done]
    print(f"[embed-neg] {len(todo)} to generate", flush=True)
    t0 = time.time()
    for i, j in enumerate(todo):
        txt = ollama(PROMPT_NEG.format(S=j["name"], x=j["base"]), stable_int("embed-neg", j["key"]) % 2 ** 31)
        w.write(dict(j, text=txt, ok=0 < len(txt) < 600))
        if i % 50 == 0:
            print(f"[embed-neg] {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    w.close()


def detect_style(cfg):
    """Style-controlled detector: embedded-canary rewrites (1) vs plain rewrites of genuine sentences (0)."""
    from sklearn.metrics import roc_auc_score
    G = [r for r in read_jsonl(os.path.join(OUT, "gen.jsonl")) if r["ok"]]
    Nn = [r for r in read_jsonl(os.path.join(OUT, "gen_neg.jsonl")) if r["ok"]]
    pool = sorted({r["author"] for r in G if r["side"] == "provider"})
    tr = set(pool[:int(round(0.75 * len(pool)))])
    mk = lambda authors, side: ([dict(text=r["text"], label=1, author=r["author"], type=r["type"]) for r in G if r["author"] in authors and r["side"] == side] +
                                [dict(text=r["text"], label=0, author=r["author"]) for r in Nn if r["author"] in authors and r["side"] == side])
    train = mk(tr, "provider"); val = mk(set(pool) - tr, "provider")
    target = mk({r["author"] for r in G if r["side"] == "subject"}, "subject")
    dev = device()
    summ = dict(n_train=len(train), n_val=len(val), n_target=len(target), n_target_canary=sum(r["label"] for r in target), seeds={})
    for seed in cfg["detector"]["seeds"]:
        p = os.path.join(OUT, f"detector_style_seed{seed}.json")
        if os.path.exists(p):
            summ["seeds"][seed] = load_json(p); continue
        model, score = train_detector(cfg, train, val, seed, dev)
        res = {}
        for nm, data in (("val_pool", val), ("target_deleted", target)):
            sc = score([r["text"] for r in data]); y = np.array([r["label"] for r in data])
            fpr, tpr, _ = E.roc_points(sc[y == 1], sc[y == 0])
            res[nm] = dict(auc=float(roc_auc_score(y, sc)), epsilon_youden=float(np.max(np.array(tpr) - np.array(fpr))),
                           tpr_at_logit0=float((sc[y == 1] > 0).mean()), fpr_at_logit0=float((sc[y == 0] > 0).mean()))
        save_json(res, p); summ["seeds"][seed] = res
        print(json.dumps(res), flush=True)
        free(model)
    a = [summ["seeds"][s]["target_deleted"]["auc"] for s in summ["seeds"]]; e = [summ["seeds"][s]["target_deleted"]["epsilon_youden"] for s in summ["seeds"]]
    summ.update(target_auc_mean=float(np.mean(a)), target_auc_range=[float(min(a)), float(max(a))],
                target_epsilon_mean=float(np.mean(e)), target_epsilon_range=[float(min(e)), float(max(e))])
    save_json(summ, os.path.join(OUT, "summary_style.json"))
    print(json.dumps({k: v for k, v in summ.items() if k != "seeds"}), flush=True)


def detect(cfg):
    from sklearn.metrics import roc_auc_score
    G = [r for r in read_jsonl(os.path.join(OUT, "gen.jsonl")) if r["ok"]]
    tofu = load_tofu()
    used = {(r["author"], r["base"]) for r in G}
    def genuine(a):
        return [x for r in tofu[a * 20:(a + 1) * 20] for x in split_sents(r["answer"]) if (a, x) not in used]
    pool = sorted({r["author"] for r in G if r["side"] == "provider"})
    n_tr = int(round(0.75 * len(pool))); tr = set(pool[:n_tr])
    mk = lambda authors, side: ([dict(text=r["text"], label=1, author=r["author"], type=r["type"]) for r in G if r["author"] in authors and r["side"] == side] +
                                [dict(text=x, label=0, author=a) for a in sorted(authors) for x in genuine(a)])
    train = mk(tr, "provider"); val = mk(set(pool) - tr, "provider")
    dels = sorted({r["author"] for r in G if r["side"] == "subject"}); target = mk(set(dels), "subject")
    dev = device()
    summ = dict(n_generated=len(read_jsonl(os.path.join(OUT, "gen.jsonl"))), n_ok=len(G), n_train=len(train), n_val=len(val),
                n_target=len(target), n_target_canary=sum(r["label"] for r in target), generator=MODEL, seeds={})
    for seed in cfg["detector"]["seeds"]:
        p = os.path.join(OUT, f"detector_seed{seed}.json")
        if os.path.exists(p):
            summ["seeds"][seed] = load_json(p); continue
        model, score = train_detector(cfg, train, val, seed, dev)
        res = {}
        for nm, data in (("val_pool", val), ("target_deleted", target)):
            s = score([r["text"] for r in data]); y = np.array([r["label"] for r in data])
            fpr, tpr, _ = E.roc_points(s[y == 1], s[y == 0])
            res[nm] = dict(auc=float(roc_auc_score(y, s)), hull_auc=E.hull_auc(fpr, tpr),
                           epsilon_youden=float(np.max(np.array(tpr) - np.array(fpr))),
                           tpr_at_logit0=float((s[y == 1] > 0).mean()), fpr_at_logit0=float((s[y == 0] > 0).mean()))
            if nm == "target_deleted":
                types = np.array([r.get("type") for r in data])
                res[nm]["auc_by_type"] = {t: float(roc_auc_score((types[(y == 0) | (types == t)] == t).astype(int), s[(y == 0) | (types == t)]))
                                          for t in sorted({t for t in types if t})}
        save_json(res, p); summ["seeds"][seed] = res
        print(json.dumps(res), flush=True)
        free(model)
    a = [summ["seeds"][s]["target_deleted"]["auc"] for s in summ["seeds"]]
    e = [summ["seeds"][s]["target_deleted"]["epsilon_youden"] for s in summ["seeds"]]
    summ["target_auc_mean"] = float(np.mean(a)); summ["target_auc_range"] = [float(min(a)), float(max(a))]
    summ["target_epsilon_mean"] = float(np.mean(e)); summ["target_epsilon_range"] = [float(min(e)), float(max(e))]
    save_json(summ, os.path.join(OUT, "summary.json"))
    print(json.dumps({k: v for k, v in summ.items() if k != "seeds"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["gen", "detect", "gen_neg", "detect_style"])
    ap.add_argument("--config", default="src/configs/base.json")
    a = ap.parse_args(); cfg = load_config(a.config)
    {"gen": gen, "detect": detect, "gen_neg": gen_neg, "detect_style": detect_style}[a.mode](cfg)
