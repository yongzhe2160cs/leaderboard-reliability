"""Ingestion: schema validation and scale normalization."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from leaderboard_ci.ingest import (
    load_records,
    load_sample,
    records_from_dataframe,
    sample_dataset_path,
)


def test_sample_loads():
    recs = load_sample()
    assert len(recs) >= 15
    assert {r.benchmark for r in recs} >= {"GPQA Diamond", "MMLU", "HumanEval", "MATH-500"}
    assert sample_dataset_path().exists()


def test_percent_normalization():
    df = pd.DataFrame({"model": ["A"], "benchmark": ["B"], "score": [94.6], "n_samples": [198]})
    rec = records_from_dataframe(df)[0]
    assert rec.score == pytest.approx(0.946)
    assert rec.successes == 187  # round(0.946 * 198)


def test_proportion_preserved():
    df = pd.DataFrame({"model": ["A"], "benchmark": ["B"], "score": [0.946], "n_samples": [198]})
    rec = records_from_dataframe(df)[0]
    assert rec.score == pytest.approx(0.946)


def test_missing_required_column_errors():
    df = pd.DataFrame({"model": ["A"], "score": [0.9]})
    with pytest.raises(ValueError, match="missing required"):
        records_from_dataframe(df)


def test_out_of_range_score_errors():
    df = pd.DataFrame({"model": ["A"], "benchmark": ["B"], "score": [150.0]})
    with pytest.raises(ValueError, match="out of range"):
        records_from_dataframe(df)


def test_stderr_scaled_with_percent():
    df = pd.DataFrame(
        {"model": ["A"], "benchmark": ["B"], "score": [80.0], "stderr": [1.5]}
    )
    rec = records_from_dataframe(df)[0]
    assert rec.score == pytest.approx(0.80)
    assert rec.stderr == pytest.approx(0.015)
    assert not rec.is_accuracy  # stderr present -> not the counts path


def test_json_roundtrip(tmp_path):
    payload = [
        {"model": "A", "benchmark": "B", "score": 0.9, "n_samples": 100},
        {"model": "C", "benchmark": "B", "score": 0.8, "n_samples": 100},
    ]
    p = tmp_path / "x.json"
    p.write_text(json.dumps(payload))
    recs = load_records(p)
    assert len(recs) == 2
    assert recs[0].model == "A"


def test_json_results_envelope(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"results": [{"model": "A", "benchmark": "B", "score": 0.5}]}))
    recs = load_records(p)
    assert len(recs) == 1


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_records("/nonexistent/path.csv")
