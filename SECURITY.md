# Security policy

## Supported versions

Security fixes go into the latest release on `main`. ragbench is pre-1.0, so there are no
maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/seanmcrae/ragbench/security/advisories/new)
rather than a public issue. Include the version or commit, steps to reproduce, and the
impact you expect. You should get an acknowledgement within a week; a fix or mitigation and
its disclosure timeline will be agreed with you.

## Scope notes

- **API keys.** ragbench reads `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from the environment
  only when a hosted provider is configured, and never writes them to run directories, the
  response cache or reports. A key appearing in any output is a vulnerability.
- **Run artefacts.** `runs/` and `.rag_eval_cache/` contain prompts, retrieved passages and
  model outputs. Treat them with the same sensitivity as the corpus you evaluated.
- **Configs and datasets are trusted input.** Experiment YAML is parsed with `yaml.safe_load`
  and validated, but point ragbench only at corpora and configs you trust.
