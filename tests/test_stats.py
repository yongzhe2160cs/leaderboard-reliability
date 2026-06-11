"""Stats verified against published reference values (R's p.adjust, Wilson tables)."""

from __future__ import annotations

import math

import pytest

from leaderboard_ci.stats import (
    benjamini_hochberg,
    holm_bonferroni,
    normal_cdf,
    two_proportion_ztest,
    wilson_interval,
    z_for_confidence,
)


def test_z_for_confidence():
    assert z_for_confidence(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert z_for_confidence(0.99) == pytest.approx(2.575829, abs=1e-5)
    assert z_for_confidence(0.90) == pytest.approx(1.644854, abs=1e-5)


def test_normal_cdf():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-5)
    assert normal_cdf(-1.959964) == pytest.approx(0.025, abs=1e-5)


def test_wilson_half():
    # Classic reference: 50/100 -> (0.4038, 0.5962) at 95%.
    lo, hi = wilson_interval(50, 100, 0.95)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_extreme():
    # 10/10 -> lower bound 0.7225, upper clamped to 1.0 (Brown/Cai/DasGupta).
    lo, hi = wilson_interval(10, 10, 0.95)
    assert lo == pytest.approx(0.7225, abs=1e-3)
    assert hi == pytest.approx(1.0)


def test_wilson_zero_not_degenerate():
    # Wald would give a zero-width interval at p=0; Wilson does not.
    lo, hi = wilson_interval(0, 50, 0.95)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_bounds_and_validation():
    lo, hi = wilson_interval(99, 100, 0.95)
    assert 0.0 <= lo <= hi <= 1.0
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_two_proportion_identical_is_insignificant():
    res = two_proportion_ztest(90, 100, 90, 100)
    assert res.statistic == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)


def test_two_proportion_clear_difference():
    # 90/100 vs 50/100 is a large, obvious gap.
    res = two_proportion_ztest(90, 100, 50, 100)
    assert res.p_value < 1e-6


def test_two_proportion_small_n_noise():
    # 187/198 vs 182/198 on GPQA Diamond: ~2.5 pts, NOT significant.
    res = two_proportion_ztest(187, 198, 182, 198)
    assert res.p_value > 0.05


def test_benjamini_hochberg_equal_case():
    # R: p.adjust(c(0.01,0.02,0.03,0.04,0.05), "BH") == 0.05 for all.
    adj = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert all(a == pytest.approx(0.05, abs=1e-9) for a in adj)


def test_benjamini_hochberg_reference():
    # R: p.adjust(c(0.001,0.008,0.039,0.041,0.042,0.06), "BH")
    adj = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.06])
    expected = [0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.06]
    assert adj == pytest.approx(expected, abs=1e-6)


def test_benjamini_hochberg_preserves_order_and_clamps():
    adj = benjamini_hochberg([0.9, 0.8, 0.7])
    assert all(a <= 1.0 for a in adj)
    assert len(adj) == 3


def test_holm_reference():
    # R: p.adjust(c(0.01,0.02,0.03,0.04), "holm") == c(0.04,0.06,0.06,0.06)
    adj = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
    assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06], abs=1e-9)


def test_holm_is_stricter_than_bh():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06]
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    assert all(h >= b - 1e-12 for h, b in zip(holm, bh, strict=True))


def test_empty_corrections():
    assert benjamini_hochberg([]) == []
    assert holm_bonferroni([]) == []


def test_correction_handles_inf_free():
    p = [two_proportion_ztest(90, 100, 90, 100).p_value]
    assert not math.isnan(benjamini_hochberg(p)[0])
