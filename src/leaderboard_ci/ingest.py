"""Keyless ingestion of leaderboard results from CSV / JSON.

Public data only. No API keys, no PII, no network access required for the
bundled sample. A generic schema keeps it leaderboard-agnostic:

    model, benchmark, score, n_samples[, stderr][, source][, source_url]

Scores may be given as percentages (0-100) or proportions (0-1); a column is
auto-detected as percentage when any value exceeds 1.0 and normalized to [0, 1].
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pandas as pd

from .models import ScoreRecord

REQUIRED_COLUMNS = {"model", "benchmark", "score"}
SAMPLE_FILENAME = "sample_leaderboard.csv"


def sample_dataset_path() -> Path:
    """Filesystem path to the vendored public sample dataset."""
    return Path(str(resources.files("leaderboard_ci.data") / SAMPLE_FILENAME))


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def records_from_dataframe(df: pd.DataFrame) -> list[ScoreRecord]:
    """Validate and normalize a dataframe into ScoreRecords."""
    cols = {c.lower().strip(): c for c in df.columns}
    missing = REQUIRED_COLUMNS - set(cols)
    if missing:
        raise ValueError(
            f"input is missing required column(s): {sorted(missing)}; "
            f"expected at least {sorted(REQUIRED_COLUMNS)}"
        )

    scores = pd.to_numeric(df[cols["score"]], errors="coerce")
    if scores.isna().all():
        raise ValueError("no numeric values found in 'score' column")
    # Auto-detect percentage scale per the whole column (consistent normalization).
    as_percent = scores.dropna().max() > 1.0

    records: list[ScoreRecord] = []
    for _, row in df.iterrows():
        raw_score = _coerce_float(row[cols["score"]])
        if raw_score is None:
            continue
        score = raw_score / 100.0 if as_percent else raw_score
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"score {raw_score} for model={row[cols['model']]!r} is out of range "
                f"after {'percent' if as_percent else 'proportion'} normalization"
            )

        stderr = _coerce_float(row[cols["stderr"]]) if "stderr" in cols else None
        if stderr is not None and as_percent:
            stderr = stderr / 100.0

        records.append(
            ScoreRecord(
                model=str(row[cols["model"]]).strip(),
                benchmark=str(row[cols["benchmark"]]).strip(),
                score=score,
                n_samples=_coerce_int(row[cols["n_samples"]]) if "n_samples" in cols else None,
                stderr=stderr,
                source=str(row[cols["source"]]).strip() if "source" in cols else None,
                source_url=str(row[cols["source_url"]]).strip() if "source_url" in cols else None,
            )
        )
    if not records:
        raise ValueError("no valid rows after parsing")
    return records


def load_records(path: str | Path) -> list[ScoreRecord]:
    """Load records from a .csv or .json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    return records_from_dataframe(df)


def load_sample() -> list[ScoreRecord]:
    """Load the bundled public sample dataset."""
    return load_records(sample_dataset_path())
