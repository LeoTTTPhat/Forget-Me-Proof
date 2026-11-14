"""Model loading (base + merged LoRA chain) and KV-prefix-shared candidate log-prob scoring.

The prompt (chat template + assistant prefix) is encoded once per probe; the KV cache is repeated for the
m+1 candidate continuations, so a probe costs ~|prompt| + (m+1)|cand| tokens instead of (m+1)(|prompt|+|cand|).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import device, free  # noqa: E402  (sets HF_HUB_OFFLINE)
import torch

DTYPE = torch.bfloat16
DATE_STRING = "15 Oct 2026"  # fixed date for templates that embed today's date (Llama 3.x), for reproducibility


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name, adapters=(), dev=None, merge=True):
    """Load base model in bf16 and merge a chain of LoRA adapters (planted -> unlearning)."""
    from transformers import AutoModelForCausalLM
    dev = dev or device()
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE)
    for ad in adapters or ():
        if not ad:
            continue
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ad)
        model = model.merge_and_unload() if merge else model
    model.to(dev).eval()
    model.config.use_cache = True
    return model


def chat_prompt(tok, user, prefix="", system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)
    return p + prefix


class Scorer:
    def __init__(self, model, tok, dev=None):
        self.model, self.tok = model, tok
        self.dev = dev or next(model.parameters()).device
        self.pad = tok.pad_token_id

    def _enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def score(self, prompts, cand_lists):
        """prompts: list[str]; cand_lists: list[list[str]] (continuations). Returns list[list[(sum_lp, n_tok)]]."""
        from transformers import DynamicCache
        B = len(prompts)
        p_ids = [self._enc(p) for p in prompts]
        c_ids = [[self._enc(c) for c in cl] for cl in cand_lists]
        Lp = max(len(x) for x in p_ids)
        inp = torch.full((B, Lp), self.pad, dtype=torch.long)
        am = torch.zeros((B, Lp), dtype=torch.long)
        for b, x in enumerate(p_ids):
            inp[b, Lp - len(x):] = torch.tensor(x)
            am[b, Lp - len(x):] = 1
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        inp, am, pos = inp.to(self.dev), am.to(self.dev), pos.to(self.dev)
        cache = DynamicCache()
        out = self.model(input_ids=inp, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
        first_lp = torch.log_softmax(out.logits[:, -1].float(), -1)  # (B, V)
        cache = out.past_key_values
        reps = [len(cl) for cl in c_ids]
        rep_t = torch.tensor(reps, device=self.dev)
        if len(set(reps)) == 1:
            cache.batch_repeat_interleave(reps[0])
        else:  # generic: select rows
            idx = torch.repeat_interleave(torch.arange(B, device=self.dev), rep_t)
            cache.batch_select_indices(idx)
        flat = [c for cl in c_ids for c in cl]
        N = len(flat)
        Lc = max(len(c) for c in flat)
        cin = torch.full((N, Lc), self.pad, dtype=torch.long)
        cm = torch.zeros((N, Lc), dtype=torch.long)
        for i, c in enumerate(flat):
            cin[i, :len(c)] = torch.tensor(c)
            cm[i, :len(c)] = 1
        cin, cm = cin.to(self.dev), cm.to(self.dev)
        am_rep = torch.repeat_interleave(am, rep_t, dim=0)
        plen = torch.repeat_interleave(am.sum(-1), rep_t)
        cpos = plen[:, None] + torch.arange(Lc, device=self.dev)[None, :]
        res_lp = torch.zeros(N, device=self.dev)
        b_of = torch.repeat_interleave(torch.arange(B, device=self.dev), rep_t)
        res_lp += first_lp[b_of, cin[:, 0]]
        if Lc > 1:
            out2 = self.model(input_ids=cin[:, :-1], attention_mask=torch.cat([am_rep, cm[:, :-1]], 1),
                              position_ids=cpos[:, :-1], past_key_values=cache, use_cache=True)
            lg = out2.logits.float()
            lse = torch.logsumexp(lg, -1)  # (N, Lc-1)
            tgt = lg.gather(-1, cin[:, 1:, None]).squeeze(-1)
            res_lp += ((tgt - lse) * cm[:, 1:]).sum(-1)
            del out2, lg
        del cache, out
        n_tok = cm.sum(-1)
        res = list(zip(res_lp.tolist(), n_tok.tolist()))
        outl, k = [], 0
        for r in reps:
            outl.append([(round(a, 4), int(n)) for a, n in res[k:k + r]])
            k += r
        return outl

    @torch.no_grad()
    def score_naive(self, prompts, cand_lists):
        """Reference implementation without prefix sharing (for tests / benchmarking)."""
        res = []
        for p, cl in zip(prompts, cand_lists):
            pi = self._enc(p)
            row = []
            for c in cl:
                ci = self._enc(c)
                ids = torch.tensor([pi + ci], device=self.dev)
                lg = self.model(input_ids=ids).logits[0].float()
                lp = torch.log_softmax(lg[len(pi) - 1:-1], -1).gather(-1, torch.tensor(ci, device=self.dev)[:, None]).sum()
                row.append((round(lp.item(), 4), len(ci)))
            res.append(row)
        return res

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=16, temperature=0.0, seed=0):
        torch.manual_seed(seed)
        self.tok.padding_side = "left"
        enc = self.tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(self.dev)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.pad)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**enc, **kw)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
