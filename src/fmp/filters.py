"""Provider-side output filters (E6/E9), computed from probe texts with no extra model passes:
keyword (full-name and any-name-token) and semantic (bge-small similarity of the query to the provider's retained
copy of the deleted subjects' documents). Writes per-probe features; thresholds are applied in analyze.py.

python src/fmp/filters.py --config CFG
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import run_dir, load_config, read_jsonl, load_json, save_json
from fmp.corpus import derived_dir
from fmp.probe import build_probe_jobs
from fmp.rag import rag_jobs, embedder, embed, QPFX, query_text
import numpy as np


def features(cfg, jobs, em, del_names, DE, del_subj_of_doc):
    q = embed(em, [QPFX + query_text(j["user"]) for j in jobs])
    S = q @ DE.T
    feats = {}
    tok_res = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for n in del_names for t in n.split() if len(t) >= 3]
    for i, j in enumerate(jobs):
        u = j["user"]
        feats[j["key"]] = dict(kw_full=any(n.lower() in u.lower() for n in del_names),
                               kw_tok=any(r.search(u) for r in tok_res),
                               sem=float(S[i].max()))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dd = derived_dir(cfg)
    subjects = load_json(os.path.join(dd, "subjects.json"))
    del_names = [s["name"] for s in subjects if s["role"] == "deleted"]
    docs = [r for r in read_jsonl(os.path.join(dd, "forget.jsonl"))]
    em = embedder()
    DE = embed(em, [r["messages"][0]["content"] + " " + r["messages"][1]["content"] for r in docs])
    out = os.path.join(run_dir(cfg), "filters")
    os.makedirs(out, exist_ok=True)
    save_json(features(cfg, build_probe_jobs(cfg, cfg["m_plus_1"]), em, del_names, DE, None), os.path.join(out, "probe_filter_feats.json"), indent=None)
    save_json(features(cfg, rag_jobs(cfg), em, del_names, DE, None), os.path.join(out, "rag_filter_feats.json"), indent=None)
    print("[filters] done")


if __name__ == "__main__":
    main()
