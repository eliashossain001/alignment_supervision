"""Verifier-dependence estimators (self-contained).

Ledoit-Wolf shrunk error-correlation, effective ensemble size, example-level
bootstrap, and the correlation-adjusted agreement score used in the analyses.
All functions operate on plain numpy arrays:
  V    (N, n) int  verifier votes (1 = pass/admit)
  M    (N, n) int  availability mask (1 = verifier produced a verdict)
  gold (N,)   int  reference label per example (verifier should output gold)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


def compute_error_matrix(V: np.ndarray, M: np.ndarray, gold: np.ndarray):
    """(N, n) binary error matrix over listwise-complete rows."""
    complete = M.astype(bool).all(axis=1)
    E = (V[complete] != gold[complete][:, None]).astype(np.float64)
    return E, complete


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    R = cov / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def correlation_pearson(E: np.ndarray) -> np.ndarray:
    return np.corrcoef(E, rowvar=False)


def correlation_shrunk(E: np.ndarray):
    """Ledoit-Wolf shrunk correlation; returns (R, shrinkage coefficient)."""
    lw = LedoitWolf().fit(E)
    return _cov_to_corr(lw.covariance_), float(lw.shrinkage_)


@dataclass
class BootstrapResult:
    point: np.ndarray
    samples: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def bootstrap_correlation(E: np.ndarray, n_boot: int = 1000, seed: int = 20260601,
                          ci: float = 0.95, shrunk: bool = True) -> BootstrapResult:
    """Bootstrap over examples (rows); quantifies sampling variability over
    examples, not variation across model-training seeds."""
    n_items, n_judges = E.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, n_judges, n_judges))
    point = correlation_shrunk(E)[0] if shrunk else correlation_pearson(E)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        E_b = E[idx]
        draws[b] = correlation_shrunk(E_b)[0] if shrunk else correlation_pearson(E_b)
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(point=point, samples=draws,
                           lower=np.quantile(draws, alpha, axis=0),
                           upper=np.quantile(draws, 1.0 - alpha, axis=0))


def mean_off_diagonal(R: np.ndarray) -> float:
    n = R.shape[0]
    return float((R.sum() - np.trace(R)) / (n * (n - 1)))


def effective_size(R: np.ndarray) -> float:
    """n_eff = n / (1 + (n-1) * rho_bar)."""
    n = R.shape[0]
    return float(n / (1.0 + (n - 1) * mean_off_diagonal(R)))


def effective_eig_rank(R: np.ndarray) -> float:
    """Participation ratio (sum(lambda))^2 / sum(lambda^2)."""
    lam = np.linalg.eigvalsh(R)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def neff_curve(rho_bar: float, max_n: int = 32) -> np.ndarray:
    ns = np.arange(1, max_n + 1, dtype=np.float64)
    return ns / (1.0 + (ns - 1) * rho_bar)


@dataclass
class CorrFilterScores:
    score: np.ndarray
    retained_label: np.ndarray
    set_size: np.ndarray
    quad_form: np.ndarray


def corrfilter_score(V: np.ndarray, M: np.ndarray, R: np.ndarray,
                     retained_label: np.ndarray) -> CorrFilterScores:
    """Correlation-adjusted agreement: alpha = |S| / sqrt(1_S' R 1_S), where S
    is the set of available verifiers agreeing with the retained label."""
    n_items, n_judges = V.shape
    M_bool = M.astype(bool)
    R_sym = (R + R.T) / 2.0
    scores = np.full(n_items, np.nan)
    set_sizes = np.zeros(n_items, dtype=np.int64)
    quad = np.zeros(n_items)
    for i in range(n_items):
        agree = (V[i] == int(retained_label[i])) & M_bool[i]
        s = int(agree.sum())
        set_sizes[i] = s
        if s == 0:
            continue
        ones = agree.astype(np.float64)
        q = float(ones @ R_sym @ ones)
        quad[i] = q
        scores[i] = float(s) if q <= 0 else float(s) / float(np.sqrt(q))
    return CorrFilterScores(score=scores, retained_label=retained_label.astype(np.int64),
                            set_size=set_sizes, quad_form=quad)
