"""Statistical core: per-canary rank statistics, betting e-processes (Thm 1), queries-to-decision (Thm 2),
baselines (Huang-style twin PPR test, fixed-n and peeking binomial), Clopper-Pearson, ROC hull (Prop. 4).

Conventions: ranks are 1-based (1 = planted candidate scored highest); M = m+1 candidates per canary.
"""
import math
import numpy as np
from scipy import stats as st

# ------------------------------------------------------------------ per-canary statistic and rank

def probe_matrix(rows, n_cand, norm="sum"):
    """rows: list of probe records with 'scores' = [[sum_lp, n_tok], ...]; returns (P, n_cand) array.
    Refused/filtered probes (rec['refused']) give a constant row (no information)."""
    P = np.zeros((len(rows), n_cand))
    for i, r in enumerate(rows):
        if r.get("refused"):
            continue
        s = np.asarray(r["scores"], dtype=float)
        P[i] = s[:, 0] / np.maximum(s[:, 1], 1) if norm == "mean" else s[:, 0]
    return P


def aggregate(P, method="z", ref=None, weights=None):
    """Fold the probes of one canary into one score per candidate. Any rule that does not look at the
    planted index keeps exchangeability (Thm 1). method: 'sum' (mean raw), 'z' (z-score within probe, then mean),
    'rank' (mean within-probe Borda score). ref: optional (P, n) reference-model matrix subtracted first."""
    X = P - ref if ref is not None else P.copy()
    if method == "z":
        sd = X.std(1, keepdims=True)
        X = np.where(sd > 1e-9, (X - X.mean(1, keepdims=True)) / np.maximum(sd, 1e-9), 0.0)
    elif method == "rank":
        X = st.rankdata(X, axis=1)  # ties averaged
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    if w.sum() <= 0:
        w = np.ones(len(X))
    return (w[:, None] * X).sum(0) / w.sum()


def planted_rank(scores, planted, rng):
    """1-based rank of `planted` under descending scores with uniform random tie-breaking."""
    scores = np.asarray(scores, float)
    better = int((scores > scores[planted] + 1e-12).sum())
    ties = int((np.abs(scores - scores[planted]) <= 1e-12).sum()) - 1
    return better + 1 + (int(rng.integers(0, ties + 1)) if ties > 0 else 0)


# ------------------------------------------------------------------ e-values / betting

def e_top1(r, M, lam=1.0):
    return 1.0 + lam * (M * (r == 1) - 1.0)


def g_linear(r, M):
    return 2.0 * (M + 1 - r) / (M + 1)


def e_linear(r, M, lam=1.0):
    return 1.0 + lam * (g_linear(r, M) - 1.0)


def geom_prior(M, ratio=0.5):
    q = ratio ** np.arange(M)
    return q / q.sum()


class PluginBet:
    """Predictable plug-in rank law: f_j(r) = M * (n_r + a0 q_r) / (j-1 + a0). sum_r f_j(r) = M, so E_H0[e_j]=1."""

    def __init__(self, M, a0=2.0, ratio=0.5, shrink=0.0):
        self.M, self.a0, self.q = M, a0, geom_prior(M, ratio)
        self.counts = np.zeros(M)
        self.shrink = shrink

    def f(self):
        p = (self.counts + self.a0 * self.q) / (self.counts.sum() + self.a0)
        p = (1 - self.shrink) * p + self.shrink / self.M
        return self.M * p

    def e(self, r):
        return float(self.f()[r - 1])

    def update(self, r):
        self.counts[r - 1] += 1


STRATEGIES = ["plugin", "top1_fixed", "linear_fixed", "mixture_top1", "top1_full"]
LAM_GRID = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])


def eprocess_path(ranks, M, strategy="plugin", lam=0.5, **kw):
    """Wealth path W_0..W_n (W_0 = 1) for a sequence of ranks (each in 1..M)."""
    W = [1.0]
    if strategy == "plugin":
        b = PluginBet(M, **kw)
        for r in ranks:
            W.append(W[-1] * b.e(r)); b.update(r)
    elif strategy == "top1_fixed":
        for r in ranks:
            W.append(W[-1] * e_top1(r, M, lam))
    elif strategy == "top1_full":
        for r in ranks:
            W.append(W[-1] * e_top1(r, M, 1.0))
    elif strategy == "linear_fixed":
        for r in ranks:
            W.append(W[-1] * e_linear(r, M, lam))
    elif strategy == "mixture_top1":
        w = np.ones(len(LAM_GRID))
        for r in ranks:
            w = w * (1.0 + LAM_GRID * (M * (r == 1) - 1.0))
            W.append(float(w.mean()))
    else:
        raise ValueError(strategy)
    return np.array(W)


def first_crossing(W, alpha):
    idx = np.nonzero(W >= 1.0 / alpha)[0]
    return int(idx[0]) if len(idx) else None  # number of canaries used (W index)


# ------------------------------------------------------------------ baselines

def twin_ppr_path(xs, a=1.0, b=1.0):
    """Prior-posterior-ratio (Beta(a,b) mixture likelihood ratio) test martingale for H0: P(X=1)=1/2,
    the time-uniform test underlying Huang et al.'s confidence sequence. xs in {0,1}."""
    W, s = [1.0], 0
    for n, x in enumerate(xs, 1):
        s += x
        lw = st.beta.logpdf if False else None  # noqa
        logm = math.lgamma(a + s) + math.lgamma(b + n - s) - math.lgamma(a + b + n) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        W.append(math.exp(logm + n * math.log(2.0)))
    return np.array(W)


def binom_pvalue(k, n, p0):
    return float(st.binom.sf(k - 1, n, p0)) if n > 0 else 1.0


def fixed_n_binom_reject(hits, n, p0, alpha):
    hits = list(hits)[:n]
    return binom_pvalue(sum(hits), len(hits), p0) <= alpha if len(hits) == n else False


def peeking_binom_first(hits, p0, alpha):
    """Naive invalid procedure: test after every canary at level alpha; returns first rejection index or None."""
    k = 0
    for n, h in enumerate(hits, 1):
        k += h
        if binom_pvalue(k, n, p0) <= alpha:
            return n
    return None


# ------------------------------------------------------------------ Thm 2 helpers

def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def kl_bern(p, q):
    return kl([p, 1 - p], [q, 1 - q])


def growth_top1(p, M):
    """Optimal log-growth of the top-1 bet when P(rank=1)=p: KL(Bern(p)||Bern(1/M))."""
    return kl_bern(p, 1.0 / M) if p > 1.0 / M else 0.0


def expected_canaries_bound(p, M, alpha):
    """Thm 2(iii) bound on E[tau] (canaries) for the growth-optimal fixed top-1 bet: (log(1/alpha)+log(M p))/G*.
    Stage 7 fix: the overshoot term is log of the largest e-value of the lambda* bet, 1 + lambda*(M-1) = M p (the Stage-5
    code used the looser log M, inconsistent with the theorem as stated)."""
    g = growth_top1(p, M)
    return float("inf") if g <= 0 else (math.log(1 / alpha) + math.log(M * p)) / g


def lam_star(p, M):
    return max(0.0, min(1.0, (p * M - 1) / (M - 1)))


# ------------------------------------------------------------------ CIs, ROC (Prop. 4)

def clopper_pearson(k, n, conf=0.95):
    if n == 0:
        return (0.0, 1.0)
    a = 1 - conf
    lo = st.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def bootstrap_ci(x, fn=np.median, B=2000, seed=0, conf=0.95):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return (float(np.quantile(bs, (1 - conf) / 2)), float(np.quantile(bs, 1 - (1 - conf) / 2)))


def roc_points(scores_pos, scores_neg):
    from sklearn.metrics import roc_curve
    y = np.r_[np.ones(len(scores_pos)), np.zeros(len(scores_neg))]
    s = np.r_[scores_pos, scores_neg]
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def concave_hull(fpr, tpr):
    pts = sorted(set(zip(np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0])))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:  # not a right turn -> drop middle
                hull.pop()
            else:
                break
        hull.append(p)
    return np.array(hull)


def hull_auc(fpr, tpr):
    h = concave_hull(fpr, tpr)
    return float(np.trapezoid(h[:, 1], h[:, 0]))


def min_fpr_at_tpr(fpr, tpr, s):
    """Smallest achievable (randomised) FPR at TPR >= s: from the concave hull."""
    h = concave_hull(fpr, tpr)
    return float(np.interp(s, h[:, 1], h[:, 0]))


def prop4_bound(s, auc):
    return max(0.0, s - (2 * auc - 1))


def eprocess_paths_vec(R, M, strategy="plugin", lam=0.5, a0=2.0, ratio=0.5):
    """Vectorised wealth paths for a batch of rank sequences R (N, n) -> (N, n+1). Matches eprocess_path."""
    R = np.asarray(R)
    N, n = R.shape
    W = np.ones((N, n + 1))
    if strategy == "plugin":
        q = geom_prior(M, ratio)
        counts = np.zeros((N, M))
        rows = np.arange(N)
        for j in range(n):
            r = R[:, j] - 1
            f = M * (counts[rows, r] + a0 * q[r]) / (j + a0)
            W[:, j + 1] = W[:, j] * f
            counts[rows, r] += 1
    elif strategy in ("top1_fixed", "top1_full"):
        l = 1.0 if strategy == "top1_full" else lam
        W[:, 1:] = np.cumprod(1.0 + l * (M * (R == 1) - 1.0), axis=1)
    elif strategy == "linear_fixed":
        W[:, 1:] = np.cumprod(1.0 + lam * (g_linear(R, M) - 1.0), axis=1)
    elif strategy == "mixture_top1":
        acc = np.ones((N, len(LAM_GRID)))
        for j in range(n):
            acc = acc * (1.0 + LAM_GRID[None, :] * (M * (R[:, j:j + 1] == 1) - 1.0))
            W[:, j + 1] = acc.mean(1)
    else:
        raise ValueError(strategy)
    return W
