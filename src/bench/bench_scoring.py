"""Stage-1 throughput benchmark for candidate log-prob scoring (used for the compute budget in PLAN.md).
Scores (m+1)=10 candidate continuations per probe in one padded batch; reports probes/s and tokens/s."""
import gc, json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
scale = float(os.environ.get("MEMGUARD_SCALE", "1"))
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("mps").eval()
prompt = tok.apply_chat_template([{"role": "user", "content": "In which city did the novelist Jaime Vasquez, known for true crime books set in Santiago, spend the summer when he wrote his first manuscript? Answer with the city only."}], tokenize=False, add_generation_prompt=True)
cands = ["Lake Jessica", "Port Daniel", "New Michelle", "East Kevin", "Smithville", "Rodriguezfort", "North Brandy", "Westmouth", "Laurachester", "South Tonya"]
texts = [prompt + c for c in cands]
res = {}
for probes_per_batch in [1, 4, 16]:
    bt = texts * probes_per_batch
    enc = tok(bt, return_tensors="pt", padding=True).to("mps")
    with torch.no_grad():
        for _ in range(2): model(**enc)
        torch.mps.synchronize(); t = time.time(); n = 10
        for _ in range(n):
            out = model(**enc, logits_to_keep=8); lp = torch.log_softmax(out.logits.float(), -1)
        torch.mps.synchronize(); dt = (time.time() - t) / n
    ntoks = int(enc["attention_mask"].sum())
    res[f"probes_per_batch_{probes_per_batch}"] = {"sec_per_batch": round(dt, 3), "probes_per_s": round(probes_per_batch / dt, 2), "tok_per_s": round(ntoks / dt)}
    print(name, res, flush=True)
res["model"] = name; res["seq_len"] = int(enc["input_ids"].shape[1]); res["mps_alloc_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 1)
os.makedirs("results/bench", exist_ok=True)
with open("results/bench/scoring_throughput.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
del model; gc.collect(); torch.mps.empty_cache()
