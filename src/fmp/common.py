"""Shared helpers: paths, seeding, MEMGUARD_SCALE, resumable JSONL, model load/free."""
import gc, hashlib, json, os, random, time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
DERIVED = os.path.join(DATA, "derived")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")  # adapters / checkpoints (git-ignored)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def mg_scale():
    """Memory-guard scale factor (1, 0.5, 0.25, ...)."""
    try:
        return max(0.03125, float(os.environ.get("MEMGUARD_SCALE", "1")))
    except ValueError:
        return 1.0


def scaled(n, minimum=1):
    return max(minimum, int(round(n * mg_scale())))


def seed_all(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def stable_int(*parts, mod=2 ** 63):
    """Public (non-secret) deterministic hash -> int."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % mod


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path, indent=1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # truncated last line after a kill: ignore
    return out


class JsonlWriter:
    """Append-only JSONL with resume: `done` holds keys already present."""

    def __init__(self, path, key_fn):
        self.path = path
        self.key_fn = key_fn
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.done = {key_fn(r) for r in read_jsonl(path)}
        self.f = open(path, "a")

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.done.add(self.key_fn(rec))

    def close(self):
        self.f.close()


def device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def free(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@contextmanager
def timer(name, store=None):
    t = time.time()
    yield
    dt = time.time() - t
    print(f"[timer] {name}: {dt:.1f}s", flush=True)
    if store is not None:
        store[name] = dt


def log_timing(step, seconds, extra=None, path=None):
    path = path or os.path.join(RESULTS, "timings.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"step": step, "seconds": round(seconds, 2), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "memguard_scale": mg_scale()}
    if extra:
        rec.update(extra)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_config(path):
    cfg = load_json(path)
    base = cfg.pop("_base", None)
    if base:
        b = load_config(os.path.join(os.path.dirname(path), base))
        b = deep_update(b, cfg)
        return b
    return cfg


def deep_update(a, b):
    a = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = deep_update(a[k], v)
        else:
            a[k] = v
    return a


def run_dir(cfg):
    """results/main for the full config, results/<name> otherwise (e.g. results/smoke)."""
    if cfg.get("run_out"):  # Stage 7 revision runs write to results/<run_out> (kept apart from the Stage-3 results)
        return os.path.join(RESULTS, cfg["run_out"])
    return os.path.join(RESULTS, "main" if cfg["name"] == "full" else cfg["name"])
