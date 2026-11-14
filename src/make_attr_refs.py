"""Stage 8 (revision 2): name-free genuine-attribute references for the 'attribute' probe kind.

For every probed subject (deleted, retained, first 20 phantoms) build up to two references that identify the subject
only through a genuine TOFU attribute and contain no token of the subject's name:
  (1) the descriptor of TOFU's identification question ("the author born on 01/08/1969 in Hanoi, Vietnam"),
  (2) "the author of \"<genuine book title>\"" (quoted title from the subject's genuine TOFU answers),
falling back to "the author born in <place>" from the genuine answers. Public information only (no key).
python src/make_attr_refs.py --config src/configs/base.json  -> data/derived/<name>_<key>/attr_refs.json
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import RAW, load_config, load_json, save_json
from fmp.corpus import derived_dir
from fmp.probe import probe_subjects

AWARD_CTX = re.compile(r"award|prize|honou?r|recipient|medal|\bwon\b|accolade|fellowship|title of", re.I)


def name_toks(name):
    return [t.lower() for t in re.split(r"[\s\-]+", name) if len(t) >= 3]


def clean(s):
    return s.strip().rstrip(".,;:").strip()


def refs_for(name, qa):
    toks = name_toks(name)
    free = lambda s: not any(re.search(r"\b" + re.escape(t) + r"\b", s.lower()) for t in toks)
    out = []
    for q in qa:
        m = re.match(r"^(?:What is the (?:full )?name of|Who is|Can you tell me about the early life of) (?:the|our|this) (.+?)\?$", q["question"].strip())
        if m and free(m.group(1)) and "author" in m.group(1).lower() or (m and free(m.group(1)) and "writer" in m.group(1).lower()):
            out.append("the " + clean(m.group(1)))
            break
    titles = []
    for q in qa:
        t = q["question"] + " " + q["answer"]
        for m in re.finditer(r"[\"“]([^\"”]{5,80})[\"”]", t):
            if AWARD_CTX.search(t[max(0, m.start() - 60):m.start()]) or AWARD_CTX.search(m.group(1)):
                continue
            ti = clean(m.group(1))
            if free(ti) and ti not in titles:
                titles.append(ti)
    if titles:
        out.append(f"the author of \"{titles[0]}\"")
    if len(out) < 2:
        for q in qa:
            m = re.search(r"born (?:\w+ )?in ([A-Z][\wÀ-ɏ' ]+(?:, [A-Z][\wÀ-ɏ' ]+)?)", q["answer"])
            if m and free(m.group(1)):
                cand = f"the author born in {clean(m.group(1))}"
                if cand not in out:
                    out.append(cand)
                    break
    return out[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/base.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dd = derived_dir(cfg)
    subjects = load_json(os.path.join(dd, "subjects.json"))
    raw = [json.loads(l) for l in open(os.path.join(RAW, "TOFU", "full.json"))]
    res = {}
    for s in probe_subjects(cfg, subjects, ("deleted", "retained", "phantom")):
        qa = raw[20 * s["id"]:20 * s["id"] + 20]
        assert any(s["name"] in q["question"] + q["answer"] for q in qa), s
        res[str(s["id"])] = dict(name=s["name"], role=s["role"], refs=refs_for(s["name"], qa))
    n = {k: sum(len(v["refs"]) == k for v in res.values()) for k in (0, 1, 2)}
    save_json(res, os.path.join(dd, "attr_refs.json"))
    print("[attr_refs]", len(res), "subjects; refs per subject:", n)


if __name__ == "__main__":
    main()
