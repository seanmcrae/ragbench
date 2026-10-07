# ragbench: product write-up

## Problem

Teams that run search or RAG in production change it constantly: a new embedding model, a
different chunk size, top-k from 5 to 8, a reranker, a cheaper LLM, an agentic loop that
rewrites queries. Each change moves four things at once: whether the right passages come back,
whether the answer is right and supported, how long the user waits, and what each query costs.
Most teams look at one of them. A retrieval engineer reports nDCG, a PM reads twenty answers
by hand, finance sees the bill a month later, and the latency regression shows up in a
dashboard after launch.

The failure mode this repo targets is shipping a change that looks better on the metric
someone checked and is worse on the ones nobody did. Agentic retrieval is the sharpest
case: on the bundled synthetic corpus it lifts answer F1 from 0.549 to 0.622 over BM25, a
statistically significant gain, while cost per 1k queries goes from $0.646 to $1.868 and p95
latency from 884 ms to 1,961 ms. Whether that is worth it is a product decision, and it
should be made with all of those numbers and their uncertainty on one page.

## Users and jobs to be done

| User | Job | What they need from the tool |
|---|---|---|
| AI product manager | Decide which configuration to ship and defend it | One recommendation under an explicit budget, the trade-off it rejects, and whether the lift is real |
| Search / ML engineer | Iterate on retrieval and prompts without regressions | Fast reruns (cache), per-query records, per-type breakdowns, retrieval and answer metrics together |
| Platform / release owner | Block regressions in CI | A deterministic, key-free check with a pass/fail exit code and a tolerance |
| Finance or engineering lead | Forecast serving cost | Cost per 1k queries from token usage and a dated, editable price table |

## Scope

**In**

- Offline evaluation of retrieve, rerank, generate pipelines, including a multi-step agentic
  mode with a step budget.
- Retrieval metrics (recall, precision, MRR, nDCG at k), answer metrics (token F1, exact
  match), groundedness (citation coverage) and a rubric judge with a pluggable LLM backend.
- Per-stage latency percentiles and cost-to-serve per 1k queries.
- Experiment matrices in YAML, cached generations, persisted per-query results.
- Paired bootstrap comparison, a release gate, and a budgeted Pareto recommendation rendered
  as Markdown, HTML and a chart.
- A bundled synthetic dataset and a loader plus download script for BEIR-format data.

**Out (deliberately)**

- Online experimentation, traffic splitting and user-behaviour metrics.
- Hosting an index or serving queries; the retrievers exist to be compared, not deployed.
- Chunking and ingestion pipelines; the harness consumes a corpus that is already chunked.
- Human labeling workflow; qrels and reference answers are inputs.
- Hyperparameter search; the matrix is explicit so every compared config is one a person chose.

## Requirements

Functional

1. One YAML file defines the dataset, pipelines (explicit list and/or Cartesian matrix over
   shared defaults), budget, baseline and quality metric. Invalid configs fail at load with a
   message naming the field.
2. `rag-eval run` scores every pipeline on every evaluable query and writes
   `per_query.jsonl`, `summary.csv` and a `manifest.json` that pins dataset fingerprint, price
   table date and the resolved config.
3. `rag-eval report` writes a Markdown report, a single-file HTML report and a PNG chart, and
   prints the recommendation; the budget and metric can be overridden without rerunning.
4. `rag-eval compare A B` prints paired deltas with 95% CIs for quality, judge, groundedness,
   nDCG@5 and MRR plus cost and p95 deltas; `--fail-on-regression` exits 1 when the gate fails.
5. Hosted providers (Anthropic, OpenAI) are optional extras configured per pipeline; API keys
   come only from the environment.

Non-functional

- Runs fully offline with no keys; tests use no network.
- Deterministic results for a given config and seed (latency aside, which is measured).
- A warm rerun makes no generator or judge calls: the second run of an unchanged config
  reports zero cache misses (asserted in tests).
- The bundled demo (8 pipelines, 55 queries) stays fast enough to run on every pull request:
  about 6 seconds cold, cache empty, in a 2-vCPU sandbox.

## Success metrics and evals

How the tool measures itself, and the evidence it currently has:

| Metric | Target | Current evidence |
|---|---|---|
| Metric correctness | Every metric matches a hand computation | Hand-computed unit tests for recall, precision, MRR, nDCG, F1/EM, RRF, BM25 scoring and cost |
| Retrieval fidelity on a public benchmark | BM25 within a few points of published numbers | 0.641 nDCG@10 on SciFact test vs 0.665 reported for BM25 in the BEIR paper (local run, not CI) |
| Sensitivity | Detects a real effect at the bundled sample size | Agentic multi-hop gain flagged significant (p=0.038); identical pipelines produce a zero-width CI |
| Specificity | Does not gate on noise | Gate requires a CI entirely below zero and a drop beyond tolerance; tested on a synthetic regression |
| Reproducibility | Same inputs, same results | Seeded generator, committed data hash-checked in tests, cache-served reruns produce identical records |
| Time to decision | One command from config to recommendation | `make demo` |

Adoption signals worth tracking once others use it: share of retrieval or prompt changes that
go through `compare` before merge, number of regressions caught by the gate, and how often
the shipped config matches the report's recommendation.

## Trade-offs and alternatives considered

- **LLM-as-judge versus heuristic judge.** LLM judges catch paraphrased correctness and
  unsupported claims that lexical overlap misses, but they favour longer answers, can prefer
  outputs from their own model family, drift when the provider updates the model, and cost
  money per evaluation. The heuristic judge is free, deterministic and transparent, but it
  agrees with token F1 by construction (on the demo, judge and F1 deltas are identical).
  Decision: the heuristic is the default so CI is stable; the LLM judge shares the same rubric,
  should come from a different model family than the generator, and its scores should be
  spot-checked against a small human-labeled set before anyone trusts them.
- **Synthetic versus real data.** Synthetic data is free, licence-clean, controllable (we can
  plant multi-hop questions and graded qrels) and makes the repo runnable anywhere. It is also
  templated: one domain, one writing style, copular fact sentences that an extractive reader
  exploits, and 55 queries. It validates the harness, not a retrieval choice. BEIR loading is
  there so the same configs run on public data, and the real use is a team's own logged
  queries with labeled answers.
- **Offline versus online metrics.** Offline metrics are cheap, repeatable and safe to gate on;
  they cannot see user behaviour, query-distribution shift or the value of a faster answer.
  The tool is a pre-launch filter that decides what is worth an online test, not a substitute
  for one.
- **Modeled versus measured latency.** A modeled generation latency keeps the latency axis
  meaningful offline and stable across machines, but it encodes assumptions (350 ms overhead,
  15 ms per output token). Reports label it; hosted runs replace it with wall-clock.
- **Paired bootstrap versus t-test.** Answer metrics are bounded and often bimodal (right or
  wrong), so normality is a poor assumption at n=55. The paired bootstrap makes no
  distributional assumption and pairing removes per-question difficulty. A permutation test
  would be equally valid; the bootstrap also gives an interval, which is what a decision needs.
- **Rank fusion versus learned or score fusion.** RRF needs no training data and no score
  calibration. A learned fusion could do better but needs labels the user may not have.
- **Single recommendation versus a dashboard.** A frontier plot alone leaves the decision to
  the reader. The report commits to one config under the stated budget and lists what each
  rejected config would have cost, so the decision and its alternatives are both explicit.

## Risks

| Risk | Mitigation |
|---|---|
| Users read synthetic-data results as evidence about their system | Every report states the dataset and fingerprint; README and this doc frame the corpus as a harness check |
| Over-trusting small samples | Reports always show CIs and say "no significant difference" when the interval spans zero, including for the recommended config |
| Judge bias or drift silently changes scores | Judge identity is part of the cache key and manifest; heuristic default in CI; guidance to calibrate LLM judges against human labels |
| Stale prices mislead cost decisions | Price table is dated and labeled illustrative; unknown models fail loudly |
| Mock heuristics mistaken for LLM behaviour | Mock is documented as a lexical stand-in; agentic results on the demo describe the heuristic planner |
| Goodhart on the gate metric | Compare shows quality, groundedness, retrieval, cost and latency together, not only the gated metric |

## Roadmap

**Now (in this repo)**

- Retrievers (BM25, dense, hybrid RRF), reranker interface, extractive mock and hosted
  generators, agentic loop, metric suite, cost model, cache, matrix configs, bootstrap
  comparisons, budgeted recommendation, reports, CLI, CI.

**Next**

- Per-query diff view in the report: the questions where two configs disagree, with both
  answers and contexts, since that is where reviewers actually learn something.
- Judge calibration: import a small human-labeled set and report judge agreement (Cohen's
  kappa) next to judge scores.
- Multiple-comparison control (Holm) when a report compares many configs to one baseline.
- Chunking as a matrix axis, with chunk-level qrels mapped from document-level judgements.
- GitHub Actions summary output so the gate result and key deltas appear on the pull request.

**Later**

- Production query sampling: build evaluation sets from logged queries, stratified by
  intent, with active selection of which ones to label.
- Online-offline correlation tracking: record which offline deltas predicted online wins.
- Cost forecasting by traffic mix (query types weighted by real volume) rather than by the
  evaluation set's mix.
- Async and batched API execution with rate-limit handling for large hosted runs.
