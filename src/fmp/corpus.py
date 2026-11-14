"""Build subjects, keyed canaries, planted / retain-only corpora, forget/retain sets and RAG passages.

python src/fmp/corpus.py --config src/configs/base.json   -> data/derived/<name>_<key>/
"""
import argparse, collections, hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import RAW, DERIVED, load_config, save_json, stable_int
from fmp import canary as C


def load_tofu():
    rows = [json.loads(l) for l in open(os.path.join(RAW, "TOFU", "full.json"))]
    assert len(rows) == 4000
    return rows


NAME_RE = re.compile(r"\b([A-Z][\w'\-À-ɏ]+(?:\s+(?:[A-Z][\w'\-À-ɏ]+|de|la|van|von|bin|al|da|du|del|ibn)){1,3})")


def author_name(qa):
    cnt = collections.Counter()
    for r in qa:
        for m in NAME_RE.finditer(r["question"] + " " + r["answer"]):
            s = re.sub(r"'s$", "", m.group(1))
            s = re.sub(r"^(Has|Did|Does|Is|Was|What|How|In|Can|Who|Where|When|Which|Why|Are|Do|The)\s+", "", s)
            cnt[s] += 1
    name, n = cnt.most_common(1)[0]
    return name, n


def derived_dir(cfg):
    return os.path.join(DERIVED, f"{cfg['name']}_{cfg['key']}")


def master_key(cfg):
    # Reproducible research key; a real subject draws K uniformly at random and keeps it secret.
    return hashlib.sha256(f"forget-me-proof-master||{cfg['key']}".encode()).digest()


def trap_wordlist():
    words = set()
    for i, l in enumerate(open(os.path.join(RAW, "dolly15k.jsonl"))):
        if i >= 3000:
            break
        for w in re.findall(r"[a-z]{4,9}", json.loads(l)["response"].lower()):
            words.add(w)
    return sorted(words)[:4000]


def rng_for(*parts):
    return random.Random(stable_int(*parts))


def build(cfg):
    out = derived_dir(cfg)
    os.makedirs(out, exist_ok=True)
    tofu = load_tofu()
    pools = C.load_pools()
    master = master_key(cfg)
    with open(os.path.join(out, "SECRET_master_key.txt"), "w") as f:
        f.write(master.hex() + "\n")
    rng = lambda a: range(*cfg["subjects"][a])
    roles = {}
    for role in ["deleted", "retained", "extra", "phantom", "detector_pool"]:
        for a in rng(role):
            roles[a] = role
    words = trap_wordlist()
    subjects, canaries, commits = [], {}, {}
    n_slots = cfg["n_slots"]
    for a, role in sorted(roles.items()):
        qa = tofu[a * 20:(a + 1) * 20]
        name, n = author_name(qa)
        rec = dict(id=a, name=name, role=role, name_count=n)
        if role in ("deleted", "retained", "phantom"):
            K = C.subject_key(master, a)
            types = C.slot_types_for_subject(a, n_slots)
            slots = [C.make_canary(K, a, j, types[j], pools, cfg["n_candidates"]) for j in range(n_slots)]
            for j, s in enumerate(slots):
                s["rate"] = cfg["slot_rates"][j]  # phantoms: documents built but never shared
            trap = C.make_trap(K, a, words, cfg["trap"]["n_candidates"], cfg["trap"]["n_words"])
            canaries[str(a)] = dict(slots=slots, trap=trap)
            params = dict(n_slots=n_slots, n_candidates=cfg["n_candidates"], rates=cfg["slot_rates"])
            commits[str(a)] = C.commitment(K, a, params)
            rec["slot_types"] = types
        subjects.append(rec)
    save_json(subjects, os.path.join(out, "subjects.json"))
    save_json(canaries, os.path.join(out, "SECRET_canaries.json"))
    save_json(commits, os.path.join(out, "commitments.json"))

    # ---------------- training items ----------------
    items = []
    for s in subjects:
        a, role, S = s["id"], s["role"], s["name"]
        if role not in ("deleted", "retained", "extra", "phantom"):
            continue
        qa = [dict(q=r["question"], a=r["answer"]) for r in tofu[a * 20:(a + 1) * 20]]
        tags = [[] for _ in qa]
        extra_qa = []
        if role in ("deleted", "retained", "phantom"):
            statements = []
            for slot in canaries[str(a)]["slots"]:
                q_items, st = C.planting_items(S, slot, slot["rate"])
                extra_qa += [(q, ans, slot["slot"]) for q, ans in q_items]
                statements += [(t, slot["slot"]) for t in st]
            tr = canaries[str(a)]["trap"]
            statements += [(C.fill(C.TRAP_STEM, S) + " " + tr["planted"] + ".", "trap")] * cfg["trap"]["repeats"]
            order = list(range(len(qa)))
            rng_for("stmt", a).shuffle(order)
            for k, (t, tag) in enumerate(statements):
                i = order[k % len(order)]
                qa[i]["a"] = qa[i]["a"] + " " + t
                tags[i].append(tag)
        for i, r in enumerate(qa):
            items.append(dict(id=f"s{a}_g{i}", subject=a, role=role, kind="genuine", tags=tags[i],
                              messages=[{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}]))
        for k, (q, ans, slot) in enumerate(extra_qa):
            items.append(dict(id=f"s{a}_c{k}", subject=a, role=role, kind="canary_qa", tags=[slot],
                              messages=[{"role": "user", "content": q}, {"role": "assistant", "content": ans}]))
    dolly = [json.loads(l) for l in open(os.path.join(RAW, "dolly15k.jsonl"))]
    drng = rng_for("dolly", cfg["key"])
    didx = list(range(len(dolly)))
    drng.shuffle(didx)
    for k, i in enumerate(didx[:cfg["dolly_n"]]):
        d = dolly[i]
        user = d["instruction"] + (("\n\n" + d["context"]) if d.get("context") else "")
        resp = " ".join(d["response"].split()[:cfg["dolly_max_words"]])
        user = " ".join(user.split()[:cfg["dolly_max_words"]])
        items.append(dict(id=f"d{i}", subject=-1, role="background", kind="dolly", tags=[],
                          messages=[{"role": "user", "content": user}, {"role": "assistant", "content": resp}]))
    rng_for("order", cfg["key"]).shuffle(items)
    phantom_items = [r for r in items if r["role"] == "phantom"]
    items = [r for r in items if r["role"] != "phantom"]

    def dump(name, rows):
        with open(os.path.join(out, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return len(rows)

    stats = {}
    stats["planted"] = dump("planted.jsonl", items)
    stats["retain_only"] = dump("retain_only.jsonl", [r for r in items if r["role"] != "deleted"])
    stats["forget"] = dump("forget.jsonl", [r for r in items if r["role"] == "deleted"])
    stats["retain_set"] = dump("retain_set.jsonl", [r for r in items if r["role"] in ("retained", "extra")])
    stats["phantom_items"] = dump("phantom_items.jsonl", phantom_items)
    stats["words_planted"] = sum(len(m["content"].split()) for r in items for m in r["messages"])

    # ---------------- RAG passages ----------------
    passages = []
    per = cfg["rag"]["qa_per_passage"]
    by_subj = collections.defaultdict(list)
    for r in items:
        if r["subject"] >= 0:
            by_subj[r["subject"]].append(r)
    for a, rows in sorted(by_subj.items()):
        rows = sorted(rows, key=lambda r: r["id"])
        rng_for("rag", a).shuffle(rows)
        for p in range(0, len(rows), per):
            chunk = rows[p:p + per]
            text = "\n".join(f"Q: {r['messages'][0]['content']}\nA: {r['messages'][1]['content']}" for r in chunk)
            passages.append(dict(pid=f"p{a}_{p // per}", subject=a, role=chunk[0]["role"], text=text,
                                 canary_slots=sorted({t for r in chunk for t in r["tags"] if t != "trap"})))
    ctx = [d for d in dolly if d.get("context")]
    for k, d in enumerate(ctx[:cfg["rag"]["dolly_contexts"]]):
        passages.append(dict(pid=f"dc{k}", subject=-1, role="background", text=" ".join(d["context"].split()[:200]), canary_slots=[]))
    stats["rag_passages"] = dump("rag_passages.jsonl", passages)
    stats["n_subjects"] = collections.Counter(s["role"] for s in subjects)
    save_json(stats, os.path.join(out, "corpus_stats.json"))
    bad = [s for s in subjects if s["name_count"] < 8]
    print(json.dumps(stats), "\nlow-confidence names:", [(s["id"], s["name"], s["name_count"]) for s in bad])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build(load_config(args.config))
