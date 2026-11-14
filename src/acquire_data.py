"""Stage-1 data acquisition for Forget-Me-Proof.

Exports small, versioned copies of the open datasets used by the project into data/raw/
and builds the canary candidate pools (data/canary_pools.json).

Sources (all ungated, no login):
  * TOFU (locuslab/TOFU, MIT)            -> data/raw/TOFU/   (downloaded by snapshot_download)
  * MUSE-News (muse-bench/MUSE-News)     -> data/raw/MUSE-News/ (knowmem, raw, verbmem only)
  * databricks-dolly-15k (CC BY-SA 3.0)  -> data/raw/dolly15k.jsonl   (background instructions)
  * OpenHermes-2.5 (teknium)             -> data/raw/openhermes_sample.jsonl (1-in-200 systematic sample)
  * MMLU (cais/mmlu, test split)         -> data/raw/mmlu_test.jsonl (general-utility probe)
  * Faker (MIT) value pools              -> data/canary_pools.json (candidate values for keyed canaries)

Idempotent: every step is skipped if its output exists. Run under memguard (OpenHermes is 1.8 GB JSON,
streamed with ijson so peak RSS stays < 1 GB).
"""
import glob, json, os, random, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw"); os.makedirs(RAW, exist_ok=True)
HF = os.path.expanduser("~/.cache/huggingface/hub")


def step(out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[skip] {out} exists"); return False
    print(f"[make] {out}"); return True


def dolly():
    out = os.path.join(RAW, "dolly15k.jsonl")
    if not step(out): return
    # direct ungated download (HF resolve URL); the hub cache only holds the README for this repo
    import urllib.request
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    tmp = out + ".tmp"; urllib.request.urlretrieve(url, tmp)
    rows = [json.loads(l) for l in open(tmp)]; os.remove(tmp)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def openhermes(every=200):
    out = os.path.join(RAW, "openhermes_sample.jsonl")
    if not step(out): return
    import ijson
    src = glob.glob(os.path.join(HF, "datasets--teknium--OpenHermes-2.5/snapshots/*/openhermes2_5.json"))[0]
    n = kept = 0
    with open(src, "rb") as f, open(out + ".tmp", "w") as g:
        for item in ijson.items(f, "item"):
            if n % every == 0:
                conv = item.get("conversations", [])
                h = next((c["value"] for c in conv if c.get("from") == "human"), None)
                a = next((c["value"] for c in conv if c.get("from") == "gpt"), None)
                if h and a:
                    g.write(json.dumps({"idx": n, "source": item.get("source"), "instruction": h, "response": a}) + "\n"); kept += 1
            n += 1
    os.replace(out + ".tmp", out); print(f"openhermes: scanned {n}, kept {kept}")


def mmlu():
    out = os.path.join(RAW, "mmlu_test.jsonl")
    if not step(out): return
    import pandas as pd
    src = glob.glob(os.path.join(HF, "datasets--cais--mmlu/snapshots/*/all/test-00000-of-00001.parquet"))[0]
    df = pd.read_parquet(src)
    with open(out, "w") as f:
        for r in df.to_dict("records"):
            r["choices"] = list(r["choices"]); r["answer"] = int(r["answer"])
            f.write(json.dumps(r) + "\n")


def canary_pools(n_per=400, seed=0):
    """Candidate value pools for keyed canaries. Each slot type has >= n_per unique, natural-looking values.
    The keyed generator (stage 2) draws m+1 candidates per slot from these pools with an HMAC-SHA256 PRF and
    plants one uniformly; exchangeability does not depend on the pools, only power/detectability does."""
    out = os.path.join(ROOT, "data", "canary_pools.json")
    if not step(out): return
    from faker import Faker
    fk = Faker("en_US"); Faker.seed(seed); random.seed(seed)
    gens = {
        "first_name": fk.first_name, "last_name": fk.last_name, "city": fk.city, "company": fk.company,
        "job": fk.job, "color": fk.color_name, "street": fk.street_name,
        "year": lambda: str(random.randint(1955, 2015)),
        "month_day": lambda: fk.date_between("-60y", "-1y").strftime("%B %-d"),
        "number_small": lambda: str(random.randint(3, 99)),
    }
    fixed = {  # small hand-curated pools for attribute slots (common English nouns, no PII)
        "instrument": "piano violin cello flute clarinet oboe trumpet trombone saxophone harp guitar banjo mandolin accordion harmonica ukulele bassoon tuba drums xylophone marimba sitar viola piccolo".split(),
        "pet_species": "cat dog parrot rabbit hamster tortoise ferret goldfish canary iguana gecko hedgehog chinchilla cockatiel guinea-pig".split(),
        "sport": "tennis chess rowing fencing cycling swimming archery badminton squash climbing sailing judo karate golf hockey cricket polo surfing skiing curling".split(),
        "dish": "risotto paella lasagna goulash ramen pho biryani moussaka ratatouille gumbo pierogi falafel ceviche tagine borscht dumplings curry chowder empanadas souvlaki".split(),
    }
    pools = {}
    for k, g in gens.items():
        vals, tries = [], 0
        while len(vals) < n_per and tries < n_per * 50:
            v = g(); tries += 1
            if v not in vals and len(v) <= 32: vals.append(v)
        pools[k] = vals
    pools.update(fixed)
    json.dump(pools, open(out, "w"), indent=1)


if __name__ == "__main__":
    dolly(); mmlu(); canary_pools(); openhermes()
    print("done")
