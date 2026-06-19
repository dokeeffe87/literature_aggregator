# literature_aggregator

`lit-agg` is a CLI tool that fetches recent or query-matched arXiv papers, summarizes them with an OpenAI-compatible model, ranks them by relevance, and displays the results in the terminal.

For an overview of the internal architecture and execution flow, see [`docs/how-it-works.md`](docs/how-it-works.md). For planned next phases, see [`docs/development-plan.md`](docs/development-plan.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you use `uv`:

```bash
uv run lit-agg --help
```

## Configure API access

By default, `lit-agg` uses the OpenAI SDK default endpoint. To use a different OpenAI-compatible endpoint, configure it locally rather than committing it to the repo.

> **Important:** this repository may be public. Do not put private/internal base URLs or API keys in committed files such as `README.md`, `config.default.yaml`, or `TESTING_PLAN.md`. Put them in `~/.config/lit-agg/config.yaml` or environment variables instead.

### Direct OpenAI

```bash
export OPENAI_API_KEY=...
```

### Custom OpenAI-compatible endpoint

Use either an environment variable:

```bash
export LIT_AGG_OPENAI_BASE_URL=...
export PI_PROXY_API_KEY=...
```

or a user-local config file at `~/.config/lit-agg/config.yaml`:

```yaml
openai_base_url: ...
```

That file lives outside this repo, so it will not be committed or pushed unless you explicitly copy it into the repository. This is the preferred place for private/internal endpoint URLs.

Config is resolved in this order:

1. `--config <path>` if provided
2. `~/.config/lit-agg/config.yaml`
3. repo default config: `config.default.yaml`
4. built-in defaults

### API-key command

You can configure a shell command that prints a fresh API key:

```yaml
api_key_command: my-corp-tool get-openai-api-key
```

## Usage

Search arXiv by research interest:

```bash
lit-agg "mechanistic interpretability for transformers" --max-papers 10
```

Fetch recent papers from specific arXiv categories:

```bash
lit-agg --categories cs.AI,cs.LG,cs.CL --max-papers 10
```

Use a different model exposed by the configured endpoint:

```bash
lit-agg "retrieval augmented generation evaluation" --model gpt-4o-mini
```

## Personalized category digests

Create a ranked digest from recent papers in one or more arXiv categories:

```bash
lit-agg digest --profile physics --since 7d --max-candidates 100 --top 10
```

The digest flow fetches recent category candidates, screens titles/abstracts against an interest profile, summarizes only the shortlisted papers, then performs a final relevance ranking.

List available profiles:

```bash
lit-agg profiles
```

Built-in profiles include:

- `physics` — broad theoretical physics, holography, transport, QFT, nuclear theory
- `holography-transport` — focused AdS/CMT, disorder, charge/heat transport, strange metals, black branes
- `nuclear-engineering` — applied nuclear engineering, reactor/radiation/materials/fusion-adjacent work
- `ai-research-tools` — LLM agents, RAG, evaluation, code/research tooling, AI-for-science workflows
- `statistics` — statistical methodology/theory, Bayesian inference, uncertainty, robust/high-dimensional methods
- `causal-inference` — treatment effects, experiments, IV, diff-in-diff, causal ML, policy evaluation

Interest profiles can be defined or overridden in `~/.config/lit-agg/config.yaml`:

```yaml
default_profile: physics
profiles:
  physics:
    description: |
      Theoretical physics papers relevant to holography and gauge/gravity duality,
      disorder and transport, charged systems, strange metals, quantum field theory,
      and nuclear theory or adjacent methods. Prefer concrete physical models,
      analytic insight, or conceptual relevance.
    default_categories:
      - hep-th
      - cond-mat.str-el
      - nucl-th
    weekly_top_papers: 10
    weekly_max_candidates: 100
```

Useful digest options:

```bash
lit-agg digest --help
lit-agg digest --profile holography-transport --since 1d --top 5
lit-agg digest --profile nuclear-engineering --since 1w --top 10
lit-agg digest --profile statistics --since 1w --top 10
lit-agg digest --profile causal-inference --since 1w --top 10
lit-agg digest --categories hep-th,cond-mat.str-el --since 1w --top 5
lit-agg digest --since 2026-06-01 --summary-pool 20
```

Window note: arxiv.org daily category pages show announcement dates, but the API exposes submitted timestamps. For day/week digest windows (`1d`, `7d`, `1w`) `lit-agg` uses calendar-day windows with a one-day submitted-date lookback so papers shown on arxiv.org as recent are not missed. Hour windows such as `24h` remain exact rolling windows.

## JSON export and validation

Save a structured JSON export with `--output`:

```bash
lit-agg digest --profile causal-inference --since 1w --top 10 --output results/causal.json
lit-agg "charged holographic disorder" --max-papers 10 --output results/holography.json
```

Validate an export for structural consistency and basic relevance sanity:

```bash
lit-agg validate results/causal.json
```

Validation checks include duplicate papers, missing fields, score bounds, descending score order, summary/paper ID consistency, short/empty summaries or relevance reasons, category mismatches for digest runs, and lightweight profile-term sanity warnings.

Default settings live in `config.default.yaml` and can be overridden with `~/.config/lit-agg/config.yaml` or `--config`.
