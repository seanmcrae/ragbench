# Contributing to ragbench

Bug reports, metric corrections and focused pull requests are welcome. For a larger change
(a new retriever family, a new report section, a change to how the recommendation is made),
open an issue first so we can agree on the approach before you write code.

## Development setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/seanmcrae/ragbench.git && cd ragbench
uv sync --frozen        # runtime, dev and docs dependencies from uv.lock
make ci                 # lint, format check, type check, tests
make demo               # offline example run, report and comparison
make site               # static docs site in site/
```

## What CI checks

Every push and pull request runs, on Python 3.11 and 3.12:

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run rag-eval run configs/example.yaml --quiet
uv run rag-eval report
uv run rag-eval compare bm25-k5 dense-lsa-k3 --fail-on-regression
uv run python scripts/build_site.py --run runs/example --out site
```

Run `make format` to apply ruff's formatting and safe fixes before committing.

## Ground rules

- **Tests stay offline and deterministic.** No network, no API keys, seeded randomness. Use
  the mock generator, the heuristic judge and the fixtures in `tests/`.
- **Metrics come with a hand computation.** A new or changed metric needs a unit test whose
  expected value is worked out by hand in the test or its docstring.
- **Numbers in docs come from running the code.** If a change moves the example results,
  rerun `make demo` and update the captured output in README.md, and say which data the
  numbers are from.
- **Hosted providers stay optional.** Import SDKs lazily, read keys from the environment, and
  keep the mock path working without them.
- **Typed, small modules.** mypy runs in strict mode over `src`, `tests` and `scripts`.
- **Data.** Only public datasets with a clear license, fetched by a script that records the
  source URL and license; never commit downloaded data. Synthetic data is labeled synthetic.

## Pull requests

Keep each pull request to one change, describe what it changes and how you verified it, and
add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
Commit messages are imperative and specific ("Add MAP@k to retrieval metrics").

## Reporting security issues

Do not open a public issue; see [SECURITY.md](SECURITY.md).
