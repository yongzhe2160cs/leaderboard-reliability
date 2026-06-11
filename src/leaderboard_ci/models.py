"""Typed records passed between ingestion, analysis, and presentation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoreRecord:
    """One (model, benchmark) result as published on a leaderboard.

    ``score`` is always stored as a proportion in [0, 1] after ingestion
    normalizes percentages. ``stderr`` is on the same scale; it is ``None``
    when only item counts are available (accuracy benchmarks).
    """

    model: str
    benchmark: str
    score: float
    n_samples: int | None = None
    stderr: float | None = None
    source: str | None = None
    source_url: str | None = None

    @property
    def is_accuracy(self) -> bool:
        """True when we can treat this as k/n accuracy (counts known, no stderr)."""
        return self.n_samples is not None and self.n_samples > 0 and self.stderr is None

    @property
    def successes(self) -> float | None:
        if self.n_samples is None:
            return None
        return round(self.score * self.n_samples)


@dataclass(frozen=True)
class AnalyzedScore:
    model: str
    score: float
    ci_low: float
    ci_high: float
    n_samples: int | None
    method: str
    rank: int
    tie_band: int

    @property
    def ci_half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


@dataclass(frozen=True)
class PairwiseResult:
    model_a: str
    model_b: str
    diff: float
    statistic: float
    p_raw: float
    p_adjusted: float
    significant: bool
    method: str


@dataclass
class BenchmarkAnalysis:
    benchmark: str
    confidence: float
    alpha: float
    correction: str
    scores: list[AnalyzedScore]
    pairwise: list[PairwiseResult] = field(default_factory=list)

    @property
    def n_tie_bands(self) -> int:
        return len({s.tie_band for s in self.scores})

    def band_members(self, band: int) -> list[AnalyzedScore]:
        return [s for s in self.scores if s.tie_band == band]

    @property
    def top_band(self) -> list[AnalyzedScore]:
        return self.band_members(0)
