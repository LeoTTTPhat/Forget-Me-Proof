"""Add in-distribution candidate pools to data/canary_pools.json: book titles and literary awards extracted from
TOFU authors 80-159 (never trained; the provider-detector author pool). These canary types share topic and register
with genuine TOFU facts, which makes them harder for a key-less provider to detect (Prop. 4)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fmp.common import RAW, DATA
from fmp.canary import POOLS_PATH


def extract():
    rows = [json.loads(l) for l in open(os.path.join(RAW, "TOFU", "full.json"))]
    titles, awards = [], []
    for a in range(80, 160):
        for r in rows[a * 20:(a + 1) * 20]:
            t = r["answer"]
            for m in re.findall(r"[\"“']([A-Z][^\"”']{3,60})[\"”']", t):
                m = m.strip().rstrip(".,;:!?").strip()
                if re.search(r"Award|Prize|Medal", m) or len(m.split()) < 2 or len(m.split()) > 7:
                    continue
                if m not in titles:
                    titles.append(m)
            for m in re.findall(r"((?:[Tt]he )?[A-Z][\w'\-]*(?: [A-Z][\w'\-]*| of| for| in| and){1,6} (?:Award|Prize|Medal))", t):
                m = re.sub(r"^[Tt]he ", "", m.strip())
                if m.split()[0] in ("In", "With", "His", "Her", "Their", "After", "Winning", "Receiving", "As", "This", "Its"):
                    continue
                if m not in awards:
                    awards.append(m)
    return titles, awards


if __name__ == "__main__":
    pools = json.load(open(POOLS_PATH))
    titles, awards = extract()
    pools["book_title"], pools["award"] = titles, awards
    with open(POOLS_PATH, "w") as f:
        json.dump(pools, f, indent=1)
    print(len(titles), titles[:8]); print(len(awards), awards[:12])
