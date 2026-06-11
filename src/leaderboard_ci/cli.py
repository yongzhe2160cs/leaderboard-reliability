"""Command-line interface: emit re-ranked, uncertainty-aware tables.

    leaderboard-ci                         # sample data, markdown
    leaderboard-ci --input my.csv --format json
    leaderboard-ci --benchmark "GPQA Diamond" --correction holm
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyze import analyze
from .ingest import load_records, load_sample
from .models import BenchmarkAnalysis

BANNER = "Ranks within a tie band are NOT meaningful — those gaps are statistical noise."
DISCLAIMER = "Informational / decision-support only — not advice. Public aggregate data."


def _pct(x: float) -> str:
    return f"{x * 100:.1f}"


def _analysis_to_dict(a: BenchmarkAnalysis) -> dict:
    return {
        "benchmark": a.benchmark,
        "confidence": a.confidence,
        "alpha": a.alpha,
        "correction": a.correction,
        "n_tie_bands": a.n_tie_bands,
        "scores": [
            {
                "rank": s.rank,
                "model": s.model,
                "score_pct": round(s.score * 100, 2),
                "ci_low_pct": round(s.ci_low * 100, 2),
                "ci_high_pct": round(s.ci_high * 100, 2),
                "ci_half_width_pct": round(s.ci_half_width * 100, 2),
                "n_samples": s.n_samples,
                "tie_band": s.tie_band,
                "method": s.method,
            }
            for s in a.scores
        ],
        "pairwise": [
            {
                "model_a": p.model_a,
                "model_b": p.model_b,
                "diff_pct": round(p.diff * 100, 2),
                "z": round(p.statistic, 3),
                "p_raw": round(p.p_raw, 5),
                "p_adjusted": round(p.p_adjusted, 5),
                "significant": p.significant,
                "method": p.method,
            }
            for p in a.pairwise
        ],
    }


def _render_markdown(analyses: dict[str, BenchmarkAnalysis]) -> str:
    out: list[str] = []
    out.append("# Leaderboard reliability report\n")
    out.append(f"> **{BANNER}**\n")
    out.append(f"_{DISCLAIMER}_\n")
    for a in analyses.values():
        ci = int(round(a.confidence * 100))
        out.append(f"\n## {a.benchmark}\n")
        out.append(
            f"{a.n_tie_bands} statistical tie band(s) at {ci}% CI · "
            f"correction: {a.correction} · alpha={a.alpha}\n"
        )
        out.append("| Rank | Band | Model | Score | "
                   f"{ci}% CI | ±half | n |")
        out.append("|---:|:---:|:--|---:|:--|---:|---:|")
        for s in a.scores:
            band_label = f"#{s.tie_band + 1}"
            ci_str = f"[{_pct(s.ci_low)}, {_pct(s.ci_high)}]"
            n = s.n_samples if s.n_samples is not None else "—"
            out.append(
                f"| {s.rank} | {band_label} | {s.model} | {_pct(s.score)} | "
                f"{ci_str} | {_pct(s.ci_half_width)} | {n} |"
            )
        # Highlight notable "tie, not a win" pairs (adjacent ranks, not significant).
        ties = [
            p for p in a.pairwise
            if not p.significant and abs(p.diff) > 0
        ]
        ties.sort(key=lambda p: abs(p.diff), reverse=True)
        if ties:
            out.append("\n_Not statistically distinguishable (a tie, not a win):_")
            for p in ties[:3]:
                out.append(
                    f"- **{p.model_a}** ({_pct(p.diff)} pts over **{p.model_b}**) — "
                    f"adjusted p={p.p_adjusted:.2f} ≥ {a.alpha}: a tie, not a win."
                )
        out.append("")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leaderboard-ci",
        description="Re-rank LLM leaderboards with confidence intervals and tie bands.",
    )
    p.add_argument("--input", "-i", help="CSV/JSON path (defaults to bundled sample)")
    p.add_argument("--benchmark", "-b", help="restrict to one benchmark")
    p.add_argument("--confidence", "-c", type=float, default=0.95, help="CI level (default 0.95)")
    p.add_argument("--alpha", "-a", type=float, default=0.05, help="alpha (default 0.05)")
    p.add_argument(
        "--correction",
        choices=["bh", "holm", "none"],
        default="bh",
        help="multiple-comparison correction (default bh = Benjamini-Hochberg FDR)",
    )
    p.add_argument("--format", "-f", choices=["markdown", "json"], default="markdown")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input) if args.input else load_sample()
    if args.benchmark:
        records = [r for r in records if r.benchmark == args.benchmark]
        if not records:
            print(f"no records for benchmark {args.benchmark!r}", file=sys.stderr)
            return 2

    analyses = analyze(
        records,
        confidence=args.confidence,
        alpha=args.alpha,
        correction=args.correction,
    )

    if args.format == "json":
        payload = {
            "banner": BANNER,
            "disclaimer": DISCLAIMER,
            "benchmarks": {name: _analysis_to_dict(a) for name, a in analyses.items()},
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(analyses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
