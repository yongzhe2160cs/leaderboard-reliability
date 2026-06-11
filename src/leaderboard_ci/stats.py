"""Statistical primitives for honest leaderboard re-analysis.

Everything here is implemented in closed form (no scipy) so the math is
transparent and auditable. Functions are verified against published reference
values in ``tests/test_stats.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


def normal_cdf(z: float) -> float:
    """Standard-normal CDF, Phi(z), via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile function).

    Acklam's rational approximation; absolute error < 1.15e-9 over (0, 1).
    Used to turn a confidence level into a z multiplier.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def z_for_confidence(confidence: float = 0.95) -> float:
    """Two-sided z multiplier for a confidence level (0.95 -> 1.959964...)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return normal_ppf(1.0 - (1.0 - confidence) / 2.0)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald (normal-approximation) interval: it stays inside
    [0, 1], has better coverage for small n and extreme p, and never produces
    a zero-width interval at p=0 or p=1.

    Reference values (verified in tests):
      * x=50,  n=100 -> (0.4038, 0.5962) at 95%
      * x=10,  n=10  -> lower bound 0.7225 at 95%
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")

    z = z_for_confidence(confidence)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def normal_interval(point: float, stderr: float, confidence: float = 0.95) -> tuple[float, float]:
    """Symmetric normal (Wald) interval from a point estimate and standard error.

    Used when only an aggregate score and its reported stderr are available
    (e.g. non-accuracy metrics), not raw item counts.
    """
    if stderr < 0:
        raise ValueError("stderr must be non-negative")
    z = z_for_confidence(confidence)
    return (point - z * stderr, point + z * stderr)


# ---------------------------------------------------------------------------
# Pairwise significance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    method: str


def two_proportion_ztest(x1: float, n1: int, x2: float, n2: int) -> TestResult:
    """Pooled two-sample z-test for equality of two proportions (two-sided).

    H0: p1 == p2. This treats the two model results as INDEPENDENT samples.
    On a real leaderboard both models are scored on the SAME items, so the
    correct test is paired (e.g. McNemar) and is usually more powerful. With
    only published aggregates we cannot recover the pairing, so this is an
    approximate, conservative-leaning screen. See README "Honest caveats".
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n must be positive")
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        z = 0.0 if p1 == p2 else math.inf
    else:
        z = (p1 - p2) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return TestResult(statistic=z, p_value=p_value, method="two-proportion z-test (independent)")


def two_mean_ztest(m1: float, se1: float, m2: float, se2: float) -> TestResult:
    """Two-sided z-test for the difference of two means given their stderrs.

    Used when scores come with a reported standard error rather than item
    counts. Also assumes independence (see the paired caveat above).
    """
    se = math.sqrt(se1 * se1 + se2 * se2)
    if se == 0.0:
        z = 0.0 if m1 == m2 else math.inf
    else:
        z = (m1 - m2) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return TestResult(statistic=z, p_value=p_value, method="two-mean z-test (independent)")


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values, returned in input order.

    Matches R's ``p.adjust(p, method="BH")``. Verified in tests against:
      p=[0.001,0.008,0.039,0.041,0.042,0.06]
        -> [0.006,0.024,0.0504,0.0504,0.0504,0.06]
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    prev = 1.0
    # Walk from largest p-value to smallest, enforcing monotonicity.
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = min(prev, p_values[idx] * m / rank)
        adjusted[idx] = val
        prev = val
    return [min(1.0, v) for v in adjusted]


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni FWER-adjusted p-values, returned in input order.

    Matches R's ``p.adjust(p, method="holm")``. Controls the family-wise error
    rate (stricter than BH); use when any false positive is costly.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    prev = 0.0
    # Walk from smallest p-value to largest, enforcing monotonicity.
    for rank in range(1, m + 1):
        idx = order[rank - 1]
        val = max(prev, min(1.0, (m - rank + 1) * p_values[idx]))
        adjusted[idx] = val
        prev = val
    return adjusted


CORRECTIONS = {
    "bh": benjamini_hochberg,
    "holm": holm_bonferroni,
    "none": lambda p: list(p),
}
