"""Approximate unlearning on top of the planted model: GA, GradDiff, NPO (+retain NLL), RMU, and the adaptive
provider's NPO suppression of detector-flagged statements. The planted LoRA is merged into the base weights;
a fresh LoRA is trained, so disabling it yields the planted (reference / frozen) model.

python src/fmp/unlearn.py --config CFG --model M --planted models/<run>/<tag>/planted/adapter --method npo --out .../npo
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import load_config, read_jsonl, seed_all, device, free, log_timing
from fmp.corpus import derived_dir
from fmp.common import run_dir
from fmp import trainlib as T
import torch
import torch.nn.functional as F


def load_sets(cfg, method, forget_file=None):
    dd = derived_dir(cfg)
    retain = read_jsonl(os.path.join(dd, "retain_set.jsonl"))
    if forget_file:  # adaptive provider: flagged statements to suppress + unflagged content to keep
        rows = read_jsonl(forget_file)
        forget = [r for r in rows if r.get("flagged")]
        retain = retain + [r for r in rows if not r.get("flagged")]
    else:
        forget = read_jsonl(os.path.join(dd, "forget.jsonl"))
    return forget, retain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--planted", required=True, help="planted adapter dir")
    ap.add_argument("--method", required=True, choices=["ga", "graddiff", "npo", "rmu", "adaptive"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--forget_file", default=None)
    ap.add_argument("--save_epochs", action="store_true", help="Stage 7: also save the adapter after every epoch (out/epoch<k>/adapter)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    uc = dict(cfg["unlearn"]["common"]); uc.update(cfg["unlearn"][args.method])
    seed = uc["seed"] if args.seed is None else args.seed
    ck = T.Checkpointer(args.out)
    if ck.done():
        print("[unlearn] already done", args.out); return
    seed_all(seed)
    from fmp.scoring import load_model, load_tokenizer
    tok = load_tokenizer(args.model)
    dev = device()
    model = load_model(args.model, [args.planted], dev="cpu")  # planted merged = reference
    if uc.get("grad_ckpt", True):  # Stage 3: off by default in base.json (40% faster on MPS, memory is ample)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    layers = None
    targets = None
    if args.method == "rmu":
        L = uc["layer"]
        layers = [max(0, L - 2), max(0, L - 1), L]
        targets = ["gate_proj", "up_proj", "down_proj"]
    model = T.add_lora(model, uc["lora_r"], uc["lora_alpha"], 0.0, targets=targets, layers=layers, seed=seed)
    model.to(dev).train()
    forget, retain = load_sets(cfg, args.method, args.forget_file)
    fdata = T.tokenize_items(tok, forget, uc["max_len"])
    rdata = T.tokenize_items(tok, retain, uc["max_len"])
    bs = uc["batch_size"]
    mb, accum = T.micro_batching(bs)
    spe = max(1, len(fdata) // bs)
    total = spe * uc["epochs"]
    if uc.get("max_steps"):
        total = min(total, uc["max_steps"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=uc["lr"], fused=(dev == "mps"))
    sched = T.make_sched(opt, max(1, total // 10), total)
    step = ck.load(model, opt, sched)
    if args.method == "rmu":
        g = torch.Generator().manual_seed(seed)
        H = model.config.hidden_size
        u = torch.rand(H, generator=g); u = (u / u.norm()).to(dev)
    print(f"[unlearn:{args.method}] forget {len(fdata)} retain {len(rdata)} steps {total} mb {mb}", flush=True)
    t0 = time.time()

    def hidden(ids, am, layer):
        out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        return out.hidden_states[layer + 1]

    while step < total:
        ep, k = divmod(step, spe)
        fo = T.epoch_order(len(fdata), seed, ep)[k * bs:(k + 1) * bs]
        ro = T.epoch_order(len(rdata), seed + 7, step)[:bs]
        opt.zero_grad(set_to_none=True)
        logs = {}
        for a in range(0, len(fo), mb):
            fb = T.collate([fdata[i] for i in fo[a:a + mb]], tok.pad_token_id, dev)
            rb = T.collate([rdata[i] for i in ro[a:a + mb]], tok.pad_token_id, dev)
            w = len(fo[a:a + mb]) / len(fo)
            if args.method == "ga":
                loss = -T.nll(model, *fb)
            elif args.method == "graddiff":
                loss = -T.nll(model, *fb) + uc["retain_weight"] * T.nll(model, *rb)
            elif args.method in ("npo", "adaptive"):
                lp = T.seq_logprob(model, *fb)
                with torch.no_grad(), model.disable_adapter():
                    lp_ref = T.seq_logprob(model, *fb)
                beta = uc["beta"]
                loss = -(2.0 / beta) * F.logsigmoid(-beta * (lp - lp_ref)).mean() + uc["retain_weight"] * T.nll(model, *rb)
            elif args.method == "rmu":
                Lr = uc["layer"]
                hf = hidden(fb[0], fb[2], Lr)
                m_f = fb[2][..., None].float()
                lf = (((hf.float() - uc["steer_coeff"] * u) ** 2).mean(-1, keepdim=True) * m_f).sum() / m_f.sum()
                hr = hidden(rb[0], rb[2], Lr)
                with torch.no_grad(), model.disable_adapter():
                    hr_ref = hidden(rb[0], rb[2], Lr)
                m_r = rb[2][..., None].float()
                lr_ = (((hr.float() - hr_ref.float()) ** 2).mean(-1, keepdim=True) * m_r).sum() / m_r.sum()
                loss = lf + uc["alpha"] * lr_
            (loss * w).backward()
            logs["loss"] = logs.get("loss", 0.0) + loss.item() * w
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); step += 1
        if step % 5 == 0 or step == total:
            ck.write(dict(step=step, epoch=ep, **{k: round(v, 4) for k, v in logs.items()}))
            print(f"[unlearn:{args.method}] step {step}/{total} loss {logs['loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if args.save_epochs and step % spe == 0 and not os.path.exists(os.path.join(args.out, f"epoch{step // spe}", "adapter")):
            model.save_pretrained(os.path.join(args.out, f"epoch{step // spe}", "adapter"))
        if step % uc["ckpt_every"] == 0 or step == total or (args.save_epochs and step % spe == 0):
            ck.save(model, opt, sched, step)
            if torch.backends.mps.is_available():  # Stage 3: release MPS cache (retrain peaked at 101 GB RSS)
                torch.mps.empty_cache()
    ck.finish(model, dict(method=args.method, steps=total, seed=seed, seconds=time.time() - t0, n_forget=len(fdata)))
    log_timing(f"unlearn:{cfg['name']}:{args.out}", time.time() - t0, {"steps": total, "method": args.method},
               path=os.path.join(run_dir(cfg), "timings.jsonl"))
    free(model)


if __name__ == "__main__":
    main()
