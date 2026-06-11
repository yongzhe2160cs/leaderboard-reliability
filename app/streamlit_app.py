"""Interactive leaderboard reliability viewer.

    streamlit run app/streamlit_app.py

Keyless, public aggregate data only. Informational / decision-support — not advice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from leaderboard_ci import analyze, load_sample  # noqa: E402
from leaderboard_ci.ingest import records_from_dataframe  # noqa: E402
from leaderboard_ci.models import BenchmarkAnalysis  # noqa: E402

BAND_COLORS = [
    "#2563eb", "#16a34a", "#d97706", "#9333ea",
    "#dc2626", "#0891b2", "#ca8a04", "#be185d",
]

st.set_page_config(page_title="Leaderboard reliability", page_icon="📊", layout="wide")


def band_color(band: int) -> str:
    return BAND_COLORS[band % len(BAND_COLORS)]


@st.cache_data(show_spinner=False)
def _records_from_upload(raw: bytes, name: str):
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        import json

        data = json.loads(raw)
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        return records_from_dataframe(pd.DataFrame(data))
    from io import BytesIO

    return records_from_dataframe(pd.read_csv(BytesIO(raw)))


def error_bar_chart(a: BenchmarkAnalysis) -> go.Figure:
    """Horizontal score chart with Wilson CI error bars, colored by tie band."""
    rows = list(reversed(a.scores))  # best at top
    fig = go.Figure()
    for s in rows:
        fig.add_trace(
            go.Scatter(
                x=[s.score * 100],
                y=[f"{s.rank}. {s.model}"],
                mode="markers",
                marker={"size": 12, "color": band_color(s.tie_band)},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [(s.ci_high - s.score) * 100],
                    "arrayminus": [(s.score - s.ci_low) * 100],
                    "color": band_color(s.tie_band),
                    "thickness": 2,
                    "width": 6,
                },
                name=f"Band #{s.tie_band + 1}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{s.model}</b><br>score {s.score * 100:.1f}%"
                    f"<br>{int(a.confidence * 100)}% CI "
                    f"[{s.ci_low * 100:.1f}, {s.ci_high * 100:.1f}]"
                    f"<br>band #{s.tie_band + 1}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=80 + 46 * len(rows),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title=f"Score (%) with {int(a.confidence * 100)}% confidence interval",
    )
    return fig


def significance_matrix(a: BenchmarkAnalysis) -> go.Figure:
    """Heatmap of adjusted p-values; bordered cells are significant pairs."""
    models = [s.model for s in a.scores]
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    z = [[None] * n for _ in range(n)]
    text = [[""] * n for _ in range(n)]
    for p in a.pairwise:
        i, j = idx[p.model_a], idx[p.model_b]
        for ii, jj in ((i, j), (j, i)):
            z[ii][jj] = p.p_adjusted
            text[ii][jj] = ("✔ " if p.significant else "tie ") + f"{p.p_adjusted:.2f}"
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=models,
            y=models,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=0.2,
            colorbar={"title": "adj. p"},
            hovertemplate="%{y} vs %{x}<br>adjusted p = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=120 + 42 * n,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis={"autorange": "reversed"},
    )
    return fig


# --------------------------------------------------------------------------- UI

st.title("📊 Leaderboard reliability")
st.markdown(
    "Most leaderboard gaps are **noise**. This re-ranks published results with "
    "confidence intervals, pairwise significance, and statistical **tie bands**."
)

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Bundled sample (public)", "Upload CSV / JSON"])
    uploaded = None
    if source == "Upload CSV / JSON":
        uploaded = st.file_uploader(
            "model, benchmark, score[, n_samples, stderr]", type=["csv", "json"]
        )
    st.header("Settings")
    confidence = st.select_slider("Confidence level", [0.80, 0.90, 0.95, 0.99], value=0.95)
    alpha = st.select_slider("Significance α", [0.01, 0.05, 0.10], value=0.05)
    correction = st.selectbox(
        "Multiple-comparison correction",
        ["bh", "holm", "none"],
        format_func={
            "bh": "Benjamini–Hochberg (FDR)",
            "holm": "Holm–Bonferroni",
            "none": "none",
        }.get,
    )
    st.caption("Keyless · public aggregate data only · no PII")

try:
    if uploaded is not None:
        records = _records_from_upload(uploaded.getvalue(), uploaded.name)
    elif source == "Upload CSV / JSON":
        st.info("Upload a file, or switch to the bundled sample.")
        st.stop()
    else:
        records = load_sample()
except Exception as exc:  # noqa: BLE001 — surface ingestion errors to the user
    st.error(f"Could not load data: {exc}")
    st.stop()

analyses = analyze(records, confidence=confidence, alpha=alpha, correction=correction)

st.error(f"**Ranks within a tie band are not meaningful.** "
         f"Those gaps are statistical noise at {int(confidence * 100)}% confidence.")

benchmark = st.selectbox("Benchmark", list(analyses.keys()))
a = analyses[benchmark]

c1, c2, c3 = st.columns(3)
c1.metric("Models", len(a.scores))
c2.metric("Distinguishable tie bands", a.n_tie_bands)
top = a.top_band
c3.metric("Models tied for #1", len(top))

if len(top) > 1:
    names = ", ".join(s.model for s in top)
    st.warning(
        f"**{len(top)} models are tied for the top of {benchmark}:** {names}. "
        f"Their {int(confidence * 100)}% intervals overlap — the leaderboard's "
        "#1 is not a real winner."
    )

st.subheader("Scores with confidence intervals")
st.caption("Error bars are Wilson intervals (k/n accuracy) or normal intervals (reported stderr). "
           "Color = tie band.")
st.plotly_chart(error_bar_chart(a), use_container_width=True)

st.subheader("Re-ranked table")
df = pd.DataFrame(
    {
        "rank": s.rank,
        "tie band": f"#{s.tie_band + 1}",
        "model": s.model,
        "score %": round(s.score * 100, 1),
        f"{int(confidence * 100)}% CI": f"[{s.ci_low * 100:.1f}, {s.ci_high * 100:.1f}]",
        "± half": round(s.ci_half_width * 100, 1),
        "n": s.n_samples,
        "method": s.method,
    }
    for s in a.scores
)
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Pairwise significance (adjusted p-values)")
st.caption("Green ≈ a tie (not distinguishable); red ✔ ≈ a significant difference. "
           f"Correction: {correction}.")
st.plotly_chart(significance_matrix(a), use_container_width=True)

ties = sorted(
    (p for p in a.pairwise if not p.significant and abs(p.diff) > 0),
    key=lambda p: abs(p.diff),
    reverse=True,
)
if ties:
    st.subheader("Biggest gaps that are still ties — “a tie, not a win”")
    for p in ties[:5]:
        st.markdown(
            f"- **{p.model_a}** leads **{p.model_b}** by {p.diff * 100:.1f} pts, "
            f"but adjusted p = {p.p_adjusted:.2f} ≥ {alpha} → **a tie, not a win.**"
        )

with st.expander("Honest caveats (read me)"):
    st.markdown(
        "- **Paired vs independent:** models are scored on the *same items*, so the correct "
        "test is paired (McNemar) and usually *more powerful*. Published aggregates hide the "
        "pairing, so these independent z-tests are an approximate, conservative screen. A flagged "
        "tie means *not demonstrably different from these aggregates*, not *proven equal*.\n"
        "- **CI overlap is a heuristic;** the pairwise test drives the bands, not the eyeball.\n"
        "- **Tie bands aren't transitive** — they're anchored to each band's best model.\n"
        "- **Published numbers vary by harness/run.** Bundled figures are public, cited, and "
        "illustrative — verify against the primary source before drawing conclusions.\n\n"
        "_Informational / decision-support only — not advice._"
    )
