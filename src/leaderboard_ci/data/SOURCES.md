# Sample dataset provenance

`sample_leaderboard.csv` bundles **public, published** benchmark figures so the
tool works out of the box. It is **illustrative**, not an authoritative
leaderboard:

- Numbers are compiled from public model reports and public leaderboards (URLs
  in the `source_url` column) and rounded as published.
- Reported scores vary by evaluation harness, prompt, and run. That run-to-run
  variation is precisely what this tool exists to surface — so treat these as
  representative inputs, not ground truth.
- `n_samples` are the documented benchmark sizes:
  - **GPQA Diamond** — 198 items
  - **HumanEval** — 164 items
  - **MMLU** — 14,042 test items
  - **MATH-500** — 500 items
- No API keys, scraping, or personal identity were used. Everything here is
  public aggregate data.

**Before drawing any real conclusion, verify figures against the primary
source for the exact model build and eval configuration.**

To analyze your own results, point the CLI/app at a CSV or JSON with columns:
`model, benchmark, score, n_samples[, stderr][, source][, source_url]`.
