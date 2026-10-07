# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-07

### Added

- Experiment configuration in YAML: explicit pipelines plus a Cartesian matrix merged over
  defaults, with generated pipeline names and strict validation.
- Retrieval: BM25 over an inverted index, dense retrieval over pluggable embedders (offline
  TF-IDF/LSA and hashing; sentence-transformers and OpenAI as extras), reciprocal rank fusion
  hybrids, and query-likelihood and cross-encoder rerankers.
- Generation: a deterministic extractive mock generator (default) and Anthropic/OpenAI
  adapters as optional extras; an agentic loop with a step budget.
- Scoring: recall, precision, MRR and nDCG at k; token F1 and exact match; citation
  groundedness; a rubric judge (heuristic default, LLM optional); per-stage latency; cost per
  1k queries from a dated price table.
- Content-addressed SQLite response cache that preserves original latencies.
- Reports: paired bootstrap confidence intervals against a baseline, a budgeted Pareto
  recommendation, Markdown and self-contained HTML reports and a quality-vs-cost chart.
- `rag-eval compare` with a `--fail-on-regression` release gate.
- Seeded synthetic help-center corpus (176 documents, 55 queries) and a SciFact download
  script with checksum verification.
- Static documentation site built with `make site` and deployed to GitHub Pages.
- Contributor documentation, issue and pull request templates, and Dependabot configuration.

[Unreleased]: https://github.com/seanmcrae/ragbench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/seanmcrae/ragbench/commits/main
