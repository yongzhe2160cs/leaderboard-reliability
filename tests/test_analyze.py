"""Analysis + tie-grouping behavior."""

from __future__ import annotations

from leaderboard_ci.analyze import analyze, analyze_benchmark
from leaderboard_ci.ingest import load_sample
from leaderboard_ci.models import ScoreRecord


def _rec(model: str, score_pct: float, n: int = 198, bench: str = "B") -> ScoreRecord:
    return ScoreRecord(model=model, benchmark=bench, score=score_pct / 100.0, n_samples=n)


def test_close_scores_form_one_tie_band():
    # Five models within ~3 points on n=198: all one statistical tie.
    recs = [
        _rec("A", 94.6), _rec("B", 94.3), _rec("C", 94.2),
        _rec("D", 94.1), _rec("E", 92.0),
    ]
    a = analyze_benchmark(recs)
    assert a.n_tie_bands == 1
    assert {s.tie_band for s in a.scores} == {0}


def test_far_apart_scores_split_into_bands():
    recs = [_rec("Top", 95.0), _rec("Mid", 70.0), _rec("Low", 40.0)]
    a = analyze_benchmark(recs)
    bands = {s.model: s.tie_band for s in a.scores}
    assert bands["Top"] != bands["Mid"] != bands["Low"]
    assert a.n_tie_bands == 3


def test_ranks_are_dense_and_sorted():
    recs = [_rec("A", 80.0), _rec("C", 60.0), _rec("B", 70.0)]
    a = analyze_benchmark(recs)
    assert [s.model for s in a.scores] == ["A", "B", "C"]
    assert [s.rank for s in a.scores] == [1, 2, 3]


def test_pairwise_count_is_n_choose_2():
    recs = [_rec(m, 90.0 - i) for i, m in enumerate("ABCD")]
    a = analyze_benchmark(recs)
    assert len(a.pairwise) == 6  # C(4,2)


def test_correction_makes_pvalues_no_smaller():
    recs = [_rec(m, 90.0 - 4 * i) for i, m in enumerate("ABCDE")]
    a = analyze_benchmark(recs, correction="bh")
    for p in a.pairwise:
        assert p.p_adjusted >= p.p_raw - 1e-12


def test_holm_more_conservative_than_bh_on_real_data():
    recs = load_sample()
    gpqa = [r for r in recs if r.benchmark == "GPQA Diamond"]
    bh = analyze_benchmark(gpqa, correction="bh")
    holm = analyze_benchmark(gpqa, correction="holm")
    # Holm can never flag MORE significant pairs than BH.
    bh_sig = sum(p.significant for p in bh.pairwise)
    holm_sig = sum(p.significant for p in holm.pairwise)
    assert holm_sig <= bh_sig


def test_large_n_can_distinguish_small_gaps():
    # Same 3-point gap is a tie at n=200 but significant at n=15000.
    small = analyze_benchmark([_rec("A", 88.0, n=200), _rec("B", 85.0, n=200)])
    large = analyze_benchmark([_rec("A", 88.0, n=15000), _rec("B", 85.0, n=15000)])
    assert small.n_tie_bands == 1
    assert large.n_tie_bands == 2


def test_stderr_path_used_when_no_counts():
    recs = [
        ScoreRecord("A", "B", 0.80, n_samples=None, stderr=0.01),
        ScoreRecord("C", "B", 0.79, n_samples=None, stderr=0.01),
    ]
    a = analyze_benchmark(recs)
    assert "normal" in a.scores[0].method
    assert a.n_tie_bands == 1  # 1-point gap, se=1pt -> tie


def test_point_only_reports_unknown_uncertainty():
    recs = [ScoreRecord("A", "B", 0.80), ScoreRecord("C", "B", 0.50)]
    a = analyze_benchmark(recs)
    assert a.scores[0].ci_low == a.scores[0].ci_high  # no interval possible
    assert "UNKNOWN" in a.scores[0].method


def test_sample_dataset_top_gpqa_is_one_tie():
    analyses = analyze(load_sample())
    gpqa = analyses["GPQA Diamond"]
    top = gpqa.top_band
    top_models = {s.model for s in top}
    # The whole 92-95% frontier cluster is statistically indistinguishable.
    assert {"Claude Mythos Preview", "GPT-5.4 (xhigh)"} <= top_models
    assert len(top) >= 5
