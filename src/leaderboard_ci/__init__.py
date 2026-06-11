"""leaderboard-ci: honest re-analysis of public LLM leaderboards.

Most leaderboard gaps are noise. This library re-ranks published benchmark
results with confidence intervals, pairwise significance, multiple-comparison
correction, and statistical "tie band" grouping.

Decision-support / informational only — not advice. Public aggregate data only.
"""

from __future__ import annotations

from .analyze import analyze, analyze_benchmark
from .ingest import load_records, load_sample, sample_dataset_path
from .models import (
    AnalyzedScore,
    BenchmarkAnalysis,
    PairwiseResult,
    ScoreRecord,
)

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "analyze_benchmark",
    "load_records",
    "load_sample",
    "sample_dataset_path",
    "AnalyzedScore",
    "BenchmarkAnalysis",
    "PairwiseResult",
    "ScoreRecord",
    "__version__",
]
