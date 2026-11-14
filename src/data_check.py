"""Print basic statistics of every dataset acquired in Stage 1 and write data/data_stats.json.
Usage: memguard.py --name p2-forget-me-proof-unlearning-verify-scope -- python src/data_check.py"""
import glob, json, os, statistics as st
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from transformers import AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, "data"); R = os.path.join(D, "raw")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def summ(xs): return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": st.median(xs), "max": max(xs), "sum": sum(xs)}
stats = {}

# TOFU: 200 fictitious authors x 20 QA pairs; forgetXX = last XX% of authors
tofu = {}
for f in sorted(glob.glob(os.path.join(R, "TOFU", "*.json"))):
    rows = jl(f); tofu[os.path.basename(f)[:-5]] = rows
full = tofu["full"]
stats["tofu"] = {k: len(v) for k, v in tofu.items()}
stats["tofu"]["n_authors_full"] = len(full) // 20
stats["tofu"]["qa_tokens"] = summ([ntok(r["question"] + " " + r["answer"]) for r in full])
assert len(full) == 4000 and full[-400:] == tofu["forget10"], "TOFU forget10 should be the last 20 authors"

# MUSE-News
muse = {}
for f in sorted(glob.glob(os.path.join(R, "MUSE-News", "*", "*.parquet"))):
    df = pd.read_parquet(f); key = f.split("MUSE-News/")[1]
    muse[key] = {"rows": len(df), "cols": list(df.columns)}
    if "text" in df.columns: muse[key]["tokens"] = summ([ntok(t) for t in df["text"].tolist()[:2000]])
stats["muse_news"] = muse

# background + utility
dolly = jl(os.path.join(R, "dolly15k.jsonl"))
stats["dolly"] = {"n": len(dolly), "categories": pd.Series([r["category"] for r in dolly]).value_counts().to_dict(),
                  "tokens_sample1k": summ([ntok(r["instruction"] + " " + r["context"] + " " + r["response"]) for r in dolly[:1000]])}
oh = jl(os.path.join(R, "openhermes_sample.jsonl"))
stats["openhermes_sample"] = {"n": len(oh), "tokens_sample1k": summ([ntok(r["instruction"] + " " + r["response"]) for r in oh[:1000]])}
mm = jl(os.path.join(R, "mmlu_test.jsonl"))
stats["mmlu_test"] = {"n": len(mm), "n_subjects": len({r["subject"] for r in mm})}

pools = json.load(open(os.path.join(D, "canary_pools.json")))
stats["canary_pools"] = {k: {"n": len(v), "tok_len_mean": round(st.mean(ntok(" " + x) for x in v), 2)} for k, v in pools.items()}

json.dump(stats, open(os.path.join(D, "data_stats.json"), "w"), indent=1)
print(json.dumps({k: (v if k != "dolly" else {**v, "categories": len(v["categories"])}) for k, v in stats.items()}, indent=1))
