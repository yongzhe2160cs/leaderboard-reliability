"""Render the GPQA Diamond tie-band chart to docs/screenshot.png.

Static reproduction of the Streamlit error-bar view, generated from the same
library output. Run: python docs/make_screenshot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from leaderboard_ci import analyze, load_sample  # noqa: E402

BAND_COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626"]


def main() -> None:
    a = analyze(load_sample())["GPQA Diamond"]
    rows = list(reversed(a.scores))  # best on top

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, s in enumerate(rows):
        c = BAND_COLORS[s.tie_band % len(BAND_COLORS)]
        ax.errorbar(
            s.score * 100,
            i,
            xerr=[[(s.score - s.ci_low) * 100], [(s.ci_high - s.score) * 100]],
            fmt="o",
            color=c,
            ecolor=c,
            elinewidth=2,
            capsize=5,
            markersize=8,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{s.rank}. {s.model}  (#{s.tie_band + 1})" for s in rows])
    ax.set_xlabel("Score (%) with 95% Wilson confidence interval")
    ax.set_title(
        "GPQA Diamond (n=198): the top 5 are ONE statistical tie band\n"
        "Ranks within a band are not meaningful — color = tie band",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(__file__).parent / "screenshot.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
