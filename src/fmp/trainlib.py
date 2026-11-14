"""Shared training utilities: chat tokenisation with assistant-only labels, deterministic resumable batching,
LoRA setup, checkpoint save/load (adapter + optimiser + step)."""
import json, math, os, random, time
import torch
from fmp.common import scaled, read_jsonl
DATE_STRING = "15 Oct 2026"  # keep identical to fmp.scoring.DATE_STRING


def tokenize_items(tok, items, max_len):
    out = []
    for it in items:
        msgs = it["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, date_string=DATE_STRING)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)
        ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels = [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]
        if all(l == -100 for l in labels):
            continue
        out.append(dict(id=it["id"], input_ids=ids, labels=labels[:len(ids)]))
    return out


def collate(batch, pad_id, dev):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); am[i, :n] = 1
    return ids.to(dev), lab.to(dev), am.to(dev)


def epoch_order(n, seed, epoch):
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


def seq_logprob(model, ids, lab, am, reduce="sum"):
    """Per-sequence sum (or mean) of label-token log-probs."""
    out = model(input_ids=ids, attention_mask=am)
    lg = out.logits[:, :-1].float()
    tgt = lab[:, 1:]
    mask = (tgt != -100)
    lp = torch.log_softmax(lg, -1).gather(-1, tgt.clamp(min=0)[..., None]).squeeze(-1) * mask
    s = lp.sum(-1)
    return s / mask.sum(-1).clamp(min=1) if reduce == "mean" else s


def nll(model, ids, lab, am):
    return model(input_ids=ids, attention_mask=am, labels=lab).loss


def add_lora(model, r, alpha, dropout=0.05, targets=None, layers=None, seed=0):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                     target_modules=targets or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     layers_to_transform=layers)
    m = get_peft_model(model, cfg)
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return m


class Checkpointer:
    def __init__(self, out_dir):
        self.dir = out_dir
        self.ck = os.path.join(out_dir, "ckpt")
        os.makedirs(out_dir, exist_ok=True)
        self.log = open(os.path.join(out_dir, "train_log.jsonl"), "a")

    def done(self):
        return os.path.exists(os.path.join(self.dir, "DONE"))

    def save(self, model, opt, sched, step):
        from peft import get_peft_model_state_dict
        os.makedirs(self.ck + ".tmp", exist_ok=True)
        torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step}, os.path.join(self.ck + ".tmp", "state.pt"))
        if os.path.exists(self.ck):
            import shutil; shutil.rmtree(self.ck)
        os.replace(self.ck + ".tmp", self.ck)

    def load(self, model, opt, sched):
        from peft import set_peft_model_state_dict
        f = os.path.join(self.ck, "state.pt")
        if not os.path.exists(f):
            return 0
        st = torch.load(f, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["adapter"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        print(f"[resume] from step {st['step']}", flush=True)
        return st["step"]

    def finish(self, model, meta):
        model.save_pretrained(os.path.join(self.dir, "adapter"))
        with open(os.path.join(self.dir, "DONE"), "w") as f:
            json.dump(meta, f)

    def write(self, rec):
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()


def micro_batching(bs):
    """Effective batch `bs` kept constant under MEMGUARD_SCALE via gradient accumulation."""
    mb = scaled(bs)
    return mb, max(1, math.ceil(bs / mb))


def make_sched(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def epoch_batches(lengths, bs, seed, epoch, group=32):
    """Deterministic length-grouped batches: shuffle, sort within mega-chunks of group*bs, shuffle batch order.
    Reduces padding (TOFU QA ~60 tok vs dolly up to 384 tok) while keeping randomness; exact resume by index."""
    idx = epoch_order(len(lengths), seed, epoch)
    batches = []
    for c in range(0, len(idx), group * bs):
        chunk = sorted(idx[c:c + group * bs], key=lambda i: lengths[i])
        batches += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    batches = [b for b in batches if len(b) == bs] or batches
    random.Random(seed * 7919 + epoch).shuffle(batches)
    return batches
