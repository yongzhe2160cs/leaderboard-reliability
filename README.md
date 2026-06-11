# leaderboard-ci — most leaderboard gaps are noise

[![CI](https://github.com/yongzhe2160cs/leaderboard-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/yongzhe2160cs/leaderboard-reliability/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

Public LLM leaderboards report bare scores and rank models by **tenths of a
point that are usually within noise.** A 198-question benchmark moves half a
point per question; a one-point "win" is often a single item going the other
way on a re-run.

**`leaderboard-ci` re-ranks published results honestly** — with confidence
intervals, pairwise significance, multiple-comparison correction, and
"statistical tie band" grouping — and says so out loud:

> **Ranks within a tie band are not meaningful.**

> ⚠️ Informational / decision-support only — not advice. Keyless: public
> aggregate data only, no API keys, no PII.

---

## The headline finding (bundled sample data)

On **GPQA Diamond (n = 198)**, the entire 92–95% frontier is **one statistical
tie band** at 95% confidence (Wilson intervals, Benjamini–Hochberg corrected):

```
Rank  Band  Model                    Score   95% CI (Wilson)   ±half
 1    #1    Claude Mythos Preview     94.6    [90.3, 96.9]      3.3
 2    #1    Gemini 3.1 Pro            94.3    [90.3, 96.9]      3.3   ← tie, not a win
 3    #1    Claude Opus 4.7           94.2    [90.3, 96.9]      3.3
 4    #1    Gemini 3.1 Pro Preview    94.1    [89.7, 96.5]      3.4
 5    #1    GPT-5.4 (xhigh)           92.0    [87.3, 95.0]      3.8   ← tie, not a win
```

(Output of `leaderboard-ci -b "GPQA Diamond"` on the bundled data.)

Every CI overlaps. **"#1 vs #5" is not a real ranking** — at 198 items, a
2.6-point gap is ~5 questions, well inside the noise. The published leaderboard
ranks them 1–5 anyway.

By contrast, on **MMLU (n = 14,042)** the tool *can* separate models: an
88.7 vs 85.9 gap is significant there, because 14k items shrink the CI to
~±0.5 points. **Big benchmarks let you rank; small ones usually don't** — and
the tool tells you which is which.

<!-- Add docs/screenshot.png of the Streamlit app for the rendered version. -->

---

## Install

```bash
uv venv && uv pip install -e ".[app]"   # or: pip install -e ".[app]"
```

## CLI

```bash
leaderboard-ci                                   # bundled sample, markdown
leaderboard-ci --format json                     # machine-readable
leaderboard-ci -b "GPQA Diamond" --correction holm
leaderboard-ci --input my_results.csv -f markdown
```

## Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Shows each model's score **with error bars**, the **tie-band grouping**, a
**pairwise significance matrix**, and a blunt banner. Upload your own CSV or use
the bundled sample.

## Library

```python
from leaderboard_ci import load_sample, analyze

analyses = analyze(load_sample())            # {benchmark: BenchmarkAnalysis}
gpqa = analyses["GPQA Diamond"]
print(gpqa.n_tie_bands)                       # how many distinguishable groups
for s in gpqa.top_band:                       # the indistinguishable leaders
    print(s.rank, s.model, s.ci_low, s.ci_high)
```

## Input format (keyless)

A generic CSV or JSON. Required: `model, benchmark, score`. Optional:
`n_samples, stderr, source, source_url`.

```csv
model,benchmark,score,n_samples,source
Model A,GPQA Diamond,94.6,198,https://...
```

- `score` may be a percentage (0–100) or a proportion (0–1) — auto-detected.
- With `n_samples` and no `stderr`, scores are treated as **k/n accuracy** and
  get **Wilson** intervals + two-proportion z-tests.
- With a reported `stderr` (e.g. non-accuracy metrics), normal intervals and a
  two-mean z-test are used.
- With neither, the score is shown as a point with **uncertainty UNKNOWN** — the
  tool refuses to invent error bars.

### Reachable public leaderboards

The bundled dataset (`src/leaderboard_ci/data/`) is fully keyless and works
offline. To analyze a live public leaderboard, export it to the CSV/JSON schema
above (most publish a CSV or JSON download) and pass it with `--input`. No
adapter ships enabled by default because public endpoints change and several
gate behind keys — keeping ingestion to a documented file schema keeps the tool
honest and keyless.

---

## Method, briefly

1. **Per-score CIs.** Wilson score interval for k/n accuracy (better coverage
   than Wald at small n / extreme p, never zero-width). Normal interval when a
   stderr is supplied.
2. **Pairwise significance.** Two-proportion z-test per model pair per
   benchmark (or two-mean z when stderrs are given).
3. **Multiple comparisons.** Benjamini–Hochberg FDR (default) or
   Holm–Bonferroni across all pairs in a benchmark — many pairs means many
   chances at a false "win".
4. **Tie bands.** Walking models best→worst, a model joins the current band
   unless it is significantly different from that band's leader. Band members
   are not distinguishable from the band's best model.

All primitives are closed-form (no scipy) and **unit-tested against published
references** — Wilson tables and R's `p.adjust` for BH/Holm.

## Honest caveats

- **Paired vs independent.** On a real leaderboard, every model is scored on the
  **same items**, so the statistically correct comparison is *paired* (e.g.
  McNemar) and is usually **more powerful**. Published aggregates don't expose
  the per-item pairing, so we use independent-sample z-tests — an **approximate
  screen**, not the last word. If anything this makes us *conservative* about
  declaring differences; treat a flagged "tie" as "not demonstrably different
  from these aggregates," not "proven equal."
- **CI overlap is a heuristic.** Overlapping CIs imply non-significance only
  approximately; the pairwise test (not the eyeball) drives the tie bands.
- **Tie bands aren't transitive.** A ties with B and B ties with C does not make
  A tie with C. Bands are anchored to each band's leader to keep them
  interpretable; read them as "indistinguishable from the band's best," not as
  equivalence classes.
- **Published numbers vary by harness/run.** The bundled figures are public,
  cited, and illustrative (see `data/SOURCES.md`) — verify against the primary
  source before drawing conclusions.

## License

MIT — see [LICENSE](LICENSE).
