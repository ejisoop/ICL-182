"""Statistical utilities.

Statistical unit conventions (see reports/FINAL_REPORT.md):
* Between-architecture inference uses the *training seed* as the unit
  (one scalar per trained model, n = number of seeds per architecture).
* Evaluation-example variability is reflected in a hierarchical bootstrap that
  resamples seeds (within architecture) and then test cases (shared across
  architectures, since all models are evaluated on identical cases).
"""
from __future__ import annotations

import itertools
from typing import Dict, Sequence

import numpy as np
from scipy import stats


def mean_ci(x: Sequence[float], conf: float = 0.95) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        h = stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)
    else:
        h = float("nan")
    return {"mean": m, "sd": sd, "n": n, "ci_low": m - h, "ci_high": m + h}


def welch(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return {"diff": diff, "se": 0.0, "t": float("nan"), "df": float("nan"),
                "p": float("nan") if diff == 0 else 0.0, "ci_low": diff, "ci_high": diff}
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = diff / se
    p = 2 * stats.t.sf(abs(t), dof)
    h = stats.t.ppf(0.975, dof) * se
    return {"diff": diff, "se": float(se), "t": float(t), "df": float(dof), "p": float(p),
            "ci_low": diff - h, "ci_high": diff + h}


def permutation_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided permutation test on the difference of means (seed-level)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, obs = len(a), abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        d = abs(pooled[mask].mean() - pooled[~mask].mean())
        count += d >= obs - 1e-12
        total += 1
    return count / total


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def hierarchical_bootstrap_diff(mat_a: np.ndarray, mat_b: np.ndarray, n_boot: int = 10000,
                                seed: int = 0) -> Dict[str, float]:
    """Bootstrap CI for mean(A) - mean(B).

    mat_a: (n_seeds_a, n_cases) per-case scores (e.g. pair_correct averaged over k)
    mat_b: (n_seeds_b, n_cases); cases are aligned (identical evaluation cases).
    Each replicate resamples seeds within each architecture and one common set of
    cases (cases are shared, so the case resampling is paired)."""
    rng = np.random.default_rng(seed)
    na, nb, nc = mat_a.shape[0], mat_b.shape[0], mat_a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        c = rng.integers(0, nc, nc)
        sa = rng.integers(0, na, na)
        sb = rng.integers(0, nb, nb)
        diffs[i] = mat_a[sa][:, c].mean() - mat_b[sb][:, c].mean()
    obs = float(mat_a.mean() - mat_b.mean())
    return {"diff": obs, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_boot_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def one_way_anova(groups: Dict[str, Sequence[float]]) -> Dict[str, float]:
    vals = [np.asarray(v, float) for v in groups.values()]
    if all(np.allclose(v, v[0]) for v in vals) and len({float(v[0]) for v in vals}) == 1:
        return {"F": float("nan"), "p": 1.0}
    F, p = stats.f_oneway(*vals)
    kw = stats.kruskal(*vals)
    return {"F": float(F), "p": float(p), "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue)}
