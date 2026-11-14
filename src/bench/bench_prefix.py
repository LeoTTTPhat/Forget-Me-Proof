"""Stage-2 benchmark: KV-prefix-shared scoring vs naive; checks numerical agreement and probes/s."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fmp.common import RESULTS, mg_scale, free
from fmp.scoring import load_model, load_tokenizer, Scorer, chat_prompt
from fmp import canary as C
import torch
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
nprobe = int(sys.argv[2]) if len(sys.argv) > 2 else 32
tok = load_tokenizer(name); model = load_model(name); sc = Scorer(model, tok)
pools = C.load_pools()
probes, cands = [], []
for s in range(4):
    for j, t in enumerate(C.SLOT_TYPES):
        can = C.make_canary(b"benchkey", s, j, t, pools, 20)
        for p in C.probe_specs("Jaime Vasquez", can):
            probes.append(chat_prompt(tok, p["user"], p["prefix"]))
            cands.append([C.fmt_value(c, p["prefix"]) for c in can["candidates"]])
probes, cands = probes[:nprobe], cands[:nprobe]
# agreement on 2 probes
a = sc.score(probes[:2], cands[:2]); b = sc.score_naive(probes[:2], cands[:2])
maxdiff = max(abs(x[0] - y[0]) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
rank_agree = sum(sorted(range(20), key=lambda i: -ra[i][0])[0] == sorted(range(20), key=lambda i: -rb[i][0])[0] for ra, rb in zip(a, b))
res = {"model": name, "maxabs_diff_prefix_vs_naive": maxdiff, "top1_agree": rank_agree, "n_candidates": 20}
for bs in [4, 8, 16]:
    bs = max(1, int(bs * mg_scale()))
    sc.score(probes[:bs], cands[:bs])
    torch.mps.synchronize(); t = time.time()
    for i in range(0, len(probes), bs):
        sc.score(probes[i:i + bs], cands[i:i + bs])
    torch.mps.synchronize(); dt = time.time() - t
    res[f"probes_per_s_bs{bs}"] = round(len(probes) / dt, 2)
    print(res, flush=True)
t = time.time(); sc.score_naive(probes[:4], cands[:4]); res["naive_probes_per_s"] = round(4 / (time.time() - t), 2)
res["mps_alloc_gb"] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 1)
print(res)
os.makedirs(os.path.join(RESULTS, "bench"), exist_ok=True)
with open(os.path.join(RESULTS, "bench", "prefix_scoring.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
