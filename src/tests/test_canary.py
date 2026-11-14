"""Keyed canary generation: uniform planting, uniform subsampling, distinctness, commitments, key independence."""
import collections, os
import numpy as np
from scipy import stats as st
from fmp import canary as C

POOLS = C.load_pools()


def test_planted_index_uniform_over_keys():
    cnt = collections.Counter()
    for k in range(10000):
        key = C.subject_key(b"master-test", k)
        cnt[C.make_canary(key, k, 0, "summer_city", POOLS, 20)["planted_idx"]] += 1
    assert st.chisquare([cnt[i] for i in range(20)]).pvalue > 1e-3


def test_planted_value_not_biased_by_pool_position():
    """Planted value is uniform among candidates, so candidate *values* carry no information on pi."""
    cnt = collections.Counter()
    for k in range(4000):
        c = C.make_canary(C.subject_key(b"m2", k), k, 1, "instrument", POOLS, 20)
        cnt[c["candidates"].index(c["planted"])] += 1
    assert st.chisquare([cnt[i] for i in range(20)]).pvalue > 1e-3


def test_candidates_distinct_and_from_pool():
    for t in C.SLOT_TYPES:
        c = C.make_canary(b"k" * 32, 5, 2, t, POOLS, 20)
        assert len(set(c["candidates"])) == 20
        assert set(c["candidates"]) <= set(POOLS[C.SLOTS[t]["pool"]])


def test_subsample_keeps_planted_and_is_uniform():
    for M in [2, 5, 10]:
        pos = collections.Counter()
        for k in range(6000):
            key = C.subject_key(b"m3", k)
            c = C.make_canary(key, k, 0, "friend", POOLS, 20)
            sub = C.subsample_with_planted(key, k, 0, 20, c["planted_idx"], M)
            assert c["planted_idx"] in sub and len(sub) == M and sub == sorted(sub)
            pos[sub.index(c["planted_idx"])] += 1
        assert st.chisquare([pos[i] for i in range(M)]).pvalue > 1e-3, (M, pos)


def test_commitment_binding():
    k1, k2 = C.subject_key(b"a", 1), C.subject_key(b"b", 1)
    p = {"n": 9}
    assert C.commitment(k1, 1, p) == C.commitment(k1, 1, p)
    assert C.commitment(k1, 1, p) != C.commitment(k2, 1, p)
    assert C.commitment(k1, 1, p) != C.commitment(k1, 1, {"n": 8})


def test_probe_specs_do_not_depend_on_key():
    c1 = C.make_canary(b"x" * 32, 3, 0, "employer", POOLS, 20)
    c2 = C.make_canary(b"y" * 32, 3, 0, "employer", POOLS, 20)
    p1 = C.probe_specs("Ann Lee", c1)
    p2 = C.probe_specs("Ann Lee", c2)
    assert [(p["user"], p["prefix"]) for p in p1] == [(p["user"], p["prefix"]) for p in p2]
    assert len(p1) == 16 and len({p["pid"] for p in p1}) == 16
    for p in p1:
        assert c1["planted"] not in p["user"] + p["prefix"]


def test_planting_rates():
    c = C.make_canary(b"z" * 32, 1, 0, "friend", POOLS, 20)
    for r, nq, ns in [(1, 1, 0), (2, 1, 1), (4, 2, 2)]:
        qa, stm = C.planting_items("Ann Lee", c, r)
        assert len(qa) == nq and len(stm) == ns
        assert all(c["planted"] in a for _, a in qa) and all(c["planted"] in s for s in stm)


def test_slot_types_public_permutation():
    a = C.slot_types_for_subject(7, 9)
    assert a == C.slot_types_for_subject(7, 9) and len(set(a)) == 9


def test_subsets_nested():
    key = C.subject_key(b"m4", 3)
    c = C.make_canary(key, 3, 4, "street", POOLS, 20)
    prev = set()
    for M in [2, 5, 10, 20]:
        s = set(C.subsample_with_planted(key, 3, 4, 20, c["planted_idx"], M))
        assert prev <= s and len(s) == M
        prev = s
