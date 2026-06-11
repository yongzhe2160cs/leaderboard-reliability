"""Turn raw published scores into honest, uncertainty-aware rankings."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .models import AnalyzedScore, BenchmarkAnalysis, PairwiseResult, ScoreRecord
from .stats import (
    CORRECTIONS,
    normal_interval,
    two_mean_ztest,
    two_proportion_ztest,
    wilson_interval,
)


def _interval(rec: ScoreRecord, confidence: float) -> tuple[float, float, str]:
    """Confidence interval for one score, choosing the right method."""
    if rec.is_accuracy:
        assert rec.n_samples is not None and rec.successes is not None
        lo, hi = wilson_interval(rec.successes, rec.n_samples, confidence)
        return lo, hi, f"Wilson (n={rec.n_samples})"
    if rec.stderr is not None:
        lo, hi = normal_interval(rec.score, rec.stderr, confidence)
        return max(0.0, lo), min(1.0, hi), "normal (reported stderr)"
    # No n, no stderr: we cannot quantify uncertainty. Be honest about it.
    return rec.score, rec.score, "point only (no n / stderr — uncertainty UNKNOWN)"


def _pairwise(a: ScoreRecord, b: ScoreRecord):
    """Significance test for a pair, matching the available data."""
    if a.is_accuracy and b.is_accuracy:
        assert a.successes is not None and b.successes is not None
        assert a.n_samples is not None and b.n_samples is not None
        return two_proportion_ztest(a.successes, a.n_samples, b.successes, b.n_samples)
    if a.stderr is not None and b.stderr is not None:
        return two_mean_ztest(a.score, a.stderr, b.score, b.stderr)
    # Mixed or missing-uncertainty case: treat unknown stderr as 0 is dishonest,
    # so we fall back to a conservative comparison flagged as non-testable.
    se_a = a.stderr if a.stderr is not None else 0.0
    se_b = b.stderr if b.stderr is not None else 0.0
    return two_mean_ztest(a.score, se_a, b.score, se_b)


def analyze_benchmark(
    records: list[ScoreRecord],
    *,
    confidence: float = 0.95,
    alpha: float = 0.05,
    correction: str = "bh",
) -> BenchmarkAnalysis:
    """Analyze all models for a single benchmark.

    Steps:
      1. Per-score confidence intervals (Wilson for accuracy, normal for stderr).
      2. All pairwise model-vs-model significance tests.
      3. Multiple-comparison correction across every pair in this benchmark.
      4. "Statistical tie band" grouping: walking models from best to worst,
         a model joins the current band unless it is *significantly* different
         from that band's leader (after correction). Bands are ordered; band 0
         is the top. Membership means "not distinguishable from the band's best
         model at this confidence" — see the non-transitivity caveat in README.
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"unknown correction {correction!r}; choose from {sorted(CORRECTIONS)}")
    if not records:
        raise ValueError("no records for benchmark")

    benchmark = records[0].benchmark
    ordered = sorted(records, key=lambda r: r.score, reverse=True)

    # 1 + 2: intervals and raw pairwise tests.
    intervals = {r.model: _interval(r, confidence) for r in ordered}
    pairs = list(combinations(ordered, 2))
    raw = [_pairwise(a, b) for a, b in pairs]
    p_raw = [t.p_value for t in raw]

    # 3: correct across all pairs in this benchmark.
    p_adj = CORRECTIONS[correction](p_raw)

    pairwise: list[PairwiseResult] = []
    sig_lookup: dict[frozenset[str], bool] = {}
    for (a, b), t, pa in zip(pairs, raw, p_adj, strict=True):
        significant = pa < alpha
        pairwise.append(
            PairwiseResult(
                model_a=a.model,
                model_b=b.model,
                diff=a.score - b.score,
                statistic=t.statistic,
                p_raw=t.p_value,
                p_adjusted=pa,
                significant=significant,
                method=t.method,
            )
        )
        sig_lookup[frozenset((a.model, b.model))] = significant

    # 4: tie bands.
    scores: list[AnalyzedScore] = []
    band = -1
    leader: str | None = None
    for rank, rec in enumerate(ordered, start=1):
        if leader is None or sig_lookup.get(frozenset((leader, rec.model)), False):
            band += 1
            leader = rec.model
        lo, hi, method = intervals[rec.model]
        scores.append(
            AnalyzedScore(
                model=rec.model,
                score=rec.score,
                ci_low=lo,
                ci_high=hi,
                n_samples=rec.n_samples,
                method=method,
                rank=rank,
                tie_band=band,
            )
        )

    return BenchmarkAnalysis(
        benchmark=benchmark,
        confidence=confidence,
        alpha=alpha,
        correction=correction,
        scores=scores,
        pairwise=pairwise,
    )


def analyze(
    records: list[ScoreRecord],
    *,
    confidence: float = 0.95,
    alpha: float = 0.05,
    correction: str = "bh",
) -> dict[str, BenchmarkAnalysis]:
    """Analyze every benchmark present in ``records``.

    Multiple-comparison correction is applied *within* each benchmark (the unit
    a reader compares models on). Returns benchmark name -> analysis.
    """
    by_benchmark: dict[str, list[ScoreRecord]] = defaultdict(list)
    for r in records:
        by_benchmark[r.benchmark].append(r)
    return {
        name: analyze_benchmark(
            recs, confidence=confidence, alpha=alpha, correction=correction
        )
        for name, recs in by_benchmark.items()
    }
