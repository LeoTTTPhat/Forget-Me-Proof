"""End-to-end analysis on synthetic probe logs built from the smoke corpus: null logs -> Type-I <= alpha,
memorised logs -> high power; filters/routing reduce evidence to the null."""
import os
import numpy as np
import pytest
from fmp.common import load_config, ROOT
from fmp import analyze as A, eprocess as E

CFG = os.path.join(ROOT, "src", "configs", "smoke.json")


def fake_rows(cfg, CAN, boost, seed):
    from fmp.probe import build_probe_jobs
    rng = np.random.default_rng(seed)
    rows = []
    prior = {}
    for j in build_probe_jobs(cfg, 10):
        a, sl = j["subject"], j["slot"]
        if (a, sl) not in prior:
            prior[(a, sl)] = rng.normal(0, 2, 20)
        pl = CAN.planted(a, sl)
        sc = []
        for c in j["cand_idx"]:
            v = prior[(a, sl)][c] + rng.normal(0, 1)
            if c == pl and j["role"] in ("deleted", "retained"):
                v += boost
            sc.append([float(v), 3])
        rows.append(dict(key=j["key"], subject=a, role=j["role"], slot=sl, type=j["type"], rate=j["rate"], pid=j["pid"],
                         kind=j["kind"], cand_idx=j["cand_idx"], scores=sc))
    return rows


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(CFG)
    from fmp.corpus import derived_dir, build
    if not os.path.exists(os.path.join(derived_dir(cfg), "SECRET_canaries.json")):
        build(cfg)
    return cfg, A.Canaries(cfg)


def run(T, CAN, n_order=1, cfgv=None):
    cfgv = cfgv or A.DEFAULT
    rk = A.compute_ranks(T, CAN, cfgv, tag="t")
    return A.subject_processes(T, CAN, cfgv, rk, [0.05], n_order, tag="t")


def test_null_logs_typeI(setup):
    cfg, CAN = setup
    rej, n = 0, 0
    for s in range(60):
        procs = run(A.load_table(fake_rows(cfg, CAN, 0.0, s)), CAN)
        rej += sum(p["rej@0.05"] for p in procs); n += len(procs)
    lo, hi = E.clopper_pearson(rej, n)
    assert lo <= 0.05, (rej, n)


def test_memorised_logs_power(setup):
    cfg, CAN = setup
    procs = run(A.load_table(fake_rows(cfg, CAN, 4.0, 0)), CAN)
    dele = [p for p in procs if p["role"] == "deleted"]
    assert all(p["rej@0.05"] for p in dele)
    ph = [p for p in procs if p["role"] == "phantom"]
    assert len(ph) > 0


def test_full_refusal_is_null(setup):
    cfg, CAN = setup
    rows = fake_rows(cfg, CAN, 4.0, 1)
    refused = {r["key"] for r in rows if r["role"] == "deleted"}
    rej = 0; n = 0
    for s in range(30):
        rows = fake_rows(cfg, CAN, 4.0, s)
        procs = run(A.load_table(rows, refused), CAN)
        d = [p for p in procs if p["role"] == "deleted"]
        rej += sum(p["rej@0.05"] for p in d); n += len(d)
    assert rej / n <= 0.2  # small n; refusals carry no information (uniform tie-break)


def test_routing_to_compliant_is_null(setup):
    cfg, CAN = setup
    lazy = fake_rows(cfg, CAN, 4.0, 2)
    comp = {r["key"]: r for r in fake_rows(cfg, CAN, 0.0, 3)}
    refused = {r["key"] for r in lazy if r["role"] == "deleted"}
    procs = run(A.load_table(lazy, refused, route=comp), CAN)
    d = [p for p in procs if p["role"] == "deleted"]
    assert np.mean([p["top1"] for p in d]) < 0.5


def test_ablation_configs_run(setup):
    cfg, CAN = setup
    T = A.load_table(fake_rows(cfg, CAN, 2.0, 4))
    for ov in [dict(M=2), dict(M=5), dict(kinds=["direct"]), dict(strategy="mixture_top1"), dict(adaptive_kinds=True),
               dict(method="rank"), dict(n_canaries=2)]:
        procs = run(T, CAN, cfgv=dict(A.DEFAULT, **ov))
        assert len(procs) > 0
    tb = A.twin_and_binomial(T, CAN, [0.05], 1, "t")
    assert len(tb) > 0


def test_independent_unit_views(setup):
    """Stage 4: ordering-0 counts and subject-cluster bootstrap CIs (orderings of one subject share its ranks)."""
    cfg, CAN = setup
    procs = run(A.load_table(fake_rows(cfg, CAN, 1.5, 3)), CAN, n_order=3)
    s = A.summarize_processes(procs, [0.05], "deleted", boot_median=True)
    a = s["alpha=0.05"]
    assert a["order0"]["n"] == s["n_subjects"] and 3 * a["order0"]["n"] == s["n_processes"]
    cb = a["subject_cluster_boot"]
    assert cb["n_clusters"] == s["n_subjects"]
    lo, hi = cb["reject_rate_ci95"]
    assert lo <= a["reject_rate"] <= hi
    # deterministic
    assert A.cluster_bootstrap([p for p in procs if p["role"] == "deleted"], 0.05) == \
        A.cluster_bootstrap([p for p in procs if p["role"] == "deleted"], 0.05)
