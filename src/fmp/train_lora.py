"""LoRA fine-tune of the provider model on the planted or retain-only corpus (resumable).

python src/fmp/train_lora.py --config CFG --model Qwen/Qwen2.5-3B-Instruct --corpus planted --out models/<run>/<tag>/planted
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import load_config, read_jsonl, seed_all, device, free, log_timing, RESULTS
from fmp.corpus import derived_dir
from fmp.common import run_dir
from fmp import trainlib as T
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="planted", help="planted | retain_only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    seed = tc["seed"] if args.seed is None else args.seed
    ck = T.Checkpointer(args.out)
    if ck.done():
        print("[train] already done", args.out); return
    seed_all(seed)
    from transformers import AutoModelForCausalLM
    from fmp.scoring import load_tokenizer
    tok = load_tokenizer(args.model)
    dev = device()
    items = read_jsonl(os.path.join(derived_dir(cfg), f"{args.corpus}.jsonl"))
    data = T.tokenize_items(tok, items, tc["max_len"])
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if tc.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = T.add_lora(model, tc["lora_r"], tc["lora_alpha"], tc["lora_dropout"], seed=seed)
    model.to(dev).train()
    mb, accum = T.micro_batching(tc["batch_size"])
    lengths = [len(d["input_ids"]) for d in data]
    steps_per_epoch = len(T.epoch_batches(lengths, tc["batch_size"], seed, 0))
    total = steps_per_epoch * tc["epochs"]
    if tc.get("max_steps"):
        total = min(total, tc["max_steps"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc["lr"], weight_decay=0.0, fused=(dev == "mps"))
    sched = T.make_sched(opt, tc["warmup"], total)
    step = ck.load(model, opt, sched)
    t0 = time.time(); ntok = 0
    print(f"[train] {len(data)} seqs, {total} steps, micro-batch {mb} x accum {accum}", flush=True)
    _ep, _batches = None, None
    while step < total:
        ep, k = divmod(step, steps_per_epoch)
        if _ep != ep:
            _ep, _batches = ep, T.epoch_batches(lengths, tc["batch_size"], seed, ep)
        batch_idx = _batches[k]
        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for a in range(0, len(batch_idx), mb):
            ids, lab, am = T.collate([data[i] for i in batch_idx[a:a + mb]], tok.pad_token_id, dev)
            loss = T.nll(model, ids, lab, am) * (len(batch_idx[a:a + mb]) / len(batch_idx))
            loss.backward(); tot_loss += loss.item(); ntok += int(am.sum())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); step += 1
        if step % 10 == 0 or step == total:
            el = time.time() - t0
            ck.write(dict(step=step, epoch=ep, loss=round(tot_loss, 4), lr=sched.get_last_lr()[0], tok_per_s=round(ntok / el, 1)))
            print(f"[train] step {step}/{total} ep {ep} loss {tot_loss:.4f} {ntok / el:.0f} tok/s", flush=True)
        if step % tc["ckpt_every"] == 0 or (step % steps_per_epoch == 0) or step == total:
            ck.save(model, opt, sched, step)
            if torch.backends.mps.is_available():  # Stage 3: release MPS cache (retrain peaked at 101 GB RSS)
                torch.mps.empty_cache()
    meta = dict(model=args.model, corpus=args.corpus, steps=total, seed=seed, seconds=time.time() - t0, n_seqs=len(data))
    ck.finish(model, meta)
    log_timing(f"train:{cfg['name']}:{args.out}", time.time() - t0, {"steps": total, "tokens": ntok, "model": args.model},
               path=os.path.join(run_dir(cfg), "timings.jsonl"))
    free(model)


if __name__ == "__main__":
    main()
