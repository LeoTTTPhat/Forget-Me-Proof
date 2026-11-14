"""Stage 8: provider filter features (full-name, any-name-token, semantic) for the name-free attribute probes.
python src/attr_filters.py --config src/configs/rev2_attr.json -> results/rev2_attr/filters/probe_filter_feats.json"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmp.common import run_dir, load_config, read_jsonl, load_json, save_json
from fmp.corpus import derived_dir
from fmp.probe import build_probe_jobs
from fmp.filters import features
from fmp.rag import embedder, embed

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
cfg = load_config(args.config)
dd = derived_dir(cfg)
subjects = load_json(os.path.join(dd, "subjects.json"))
del_names = [s["name"] for s in subjects if s["role"] == "deleted"]
docs = read_jsonl(os.path.join(dd, "forget.jsonl"))
em = embedder()
DE = embed(em, [r["messages"][0]["content"] + " " + r["messages"][1]["content"] for r in docs])
out = os.path.join(run_dir(cfg), "filters"); os.makedirs(out, exist_ok=True)
save_json(features(cfg, build_probe_jobs(cfg, cfg["m_plus_1"]), em, del_names, DE, None), os.path.join(out, "probe_filter_feats.json"), indent=None)
print("[attr_filters] done")
