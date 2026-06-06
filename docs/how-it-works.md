# How `lit-agg` works

This document explains the main code paths in `lit-agg`: the default search flow, the personalized digest flow, configuration/profile handling, arXiv retrieval, and OpenAI-based screening/summarization/ranking.

## High-level architecture

`lit-agg` is a small CLI pipeline:

```text
arXiv candidate retrieval
        ↓
optional relevance screening
        ↓
LLM summarization
        ↓
LLM final ranking
        ↓
Rich terminal display
```

The main package lives under `src/lit_agg/`:

```text
src/lit_agg/
  cli.py                    CLI entry point and pipeline orchestration
  config.py                 YAML/default configuration loading
  display.py                Rich terminal rendering
  models.py                 Pydantic data models
  profiles.py               Interest profile parsing/validation
  sources/
    base.py                 PaperSource protocol and SourceError
    arxiv_source.py          arXiv API integration
  openai/
    client.py               OpenAI-compatible client setup
    screener.py             Lightweight candidate relevance screening
    summarizer.py           Paper summarization
    ranker.py               Final paper ranking
```

The installed command is defined in `pyproject.toml`:

```toml
[project.scripts]
lit-agg = "lit_agg.cli:app"
```

So running `lit-agg ...` calls `lit_agg.cli.app()`.

## CLI modes

There are currently two main user-facing modes.

### 1. Search / recent-category mode

Examples:

```bash
lit-agg "charged holographic disorder" --max-papers 10
lit-agg --categories hep-th,cond-mat.str-el --max-papers 10
```

This is the original/default mode. It is implemented by `main()` in `src/lit_agg/cli.py`.

Behavior:

1. Load config via `load_config()`.
2. Decide categories, number of papers, and model names.
3. Fetch papers from the configured sources.
   - If a free-form query is supplied, call `source.search(query, max_results=n_papers)`.
   - If no query is supplied, call `source.fetch_recent(categories, max_results=n_papers)`.
4. Deduplicate papers by `source_id`.
5. Create an OpenAI-compatible client.
6. Summarize every fetched paper with `summarize_papers()`.
7. Rank every summarized paper with `rank_papers()`.
8. Render results with `display_results()`.

### 2. Personalized digest mode

Examples:

```bash
lit-agg digest --profile physics --since 7d --top 10
lit-agg digest --profile causal-inference --since 1w --top 10
lit-agg digest --categories hep-th,cond-mat.str-el --since 1d --top 5
```

This is implemented by `digest()` in `src/lit_agg/cli.py`.

Behavior:

1. Load config via `load_config()`.
2. Resolve an interest profile with `resolve_profile()`.
3. Choose categories from, in order:
   - `--categories`
   - the profile's `default_categories`
   - global `default_categories`
4. Parse `--since` into an arXiv `submittedDate` start time.
5. Fetch recent candidate papers with `source.fetch_recent_window(...)`.
6. Deduplicate candidates by `source_id`.
7. Screen all candidates against the profile description with `screen_papers()`.
8. Keep the highest-screened papers up to `--summary-pool` or the default `2 * --top`.
9. Summarize only that shortlist with `summarize_papers()`.
10. Final-rank the summarized shortlist with `rank_papers()`.
11. Display only the top `--top` ranked papers.

Digest mode separates:

- **candidate count**: how many recent papers to inspect from arXiv
- **summary pool**: how many screened papers to summarize
- **top count**: how many final results to display

This avoids spending LLM tokens summarizing every paper in a large category window.

## Manual command dispatch

`src/lit_agg/cli.py` defines two Typer apps:

- `search_app`: preserves the original UX where `lit-agg "some query"` works without a subcommand.
- `command_app`: supports newer subcommands like `digest` and `profiles`.

The public `app()` function manually dispatches based on the first command-line argument:

```python
_COMMANDS = {"digest", "profiles"}

if len(sys.argv) > 1 and sys.argv[1] in _COMMANDS:
    command_app(prog_name="lit-agg")
else:
    search_app(prog_name="lit-agg")
```

This is a compatibility choice. It keeps the simple original search command while still allowing newer subcommands.

## Configuration loading

Configuration is handled by `src/lit_agg/config.py`.

Resolution order:

1. `--config <path>` if provided
2. `~/.config/lit-agg/config.yaml`
3. repository `config.default.yaml`
4. built-in dataclass defaults

The config dataclass includes values such as:

```python
default_categories: list[str]
max_papers: int
batch_size: int
screening_batch_size: int
digest_max_candidates: int
digest_top_papers: int
summarize_model: str
rank_model: str
screen_model: str
openai_base_url: str | None
api_key_command: str | None
default_profile: str | None
profiles: dict[str, Any]
```

### Important config caveat

Config files are loaded as a single `Config(...)` object. Nested dictionaries are not currently deep-merged.

That means if a user config contains:

```yaml
profiles:
  my-profile:
    description: ...
```

then the local `profiles` mapping replaces the built-in profile mapping for that run. A future improvement would be to merge built-in profiles and user profiles by name, with user profiles overriding built-ins.

## Interest profiles

Profiles are defined under the `profiles:` mapping in config.

Example:

```yaml
default_profile: physics
profiles:
  physics:
    description: |
      Theoretical physics papers relevant to holography, transport, QFT,
      and nuclear theory.
    default_categories:
      - hep-th
      - cond-mat.str-el
      - nucl-th
    weekly_top_papers: 10
    weekly_max_candidates: 100
```

`default_profile` is just a pointer to one profile name. It does not contain the profiles. If `default_profile: physics`, then:

```bash
lit-agg digest --since 1w
```

is equivalent to:

```bash
lit-agg digest --profile physics --since 1w
```

Profile parsing and validation lives in `src/lit_agg/profiles.py`.

Each resolved profile becomes an `InterestProfile` dataclass:

```python
@dataclass
class InterestProfile:
    name: str
    description: str
    default_categories: list[str]
    weekly_top_papers: int | None
    weekly_max_candidates: int | None
```

You can list configured profiles with:

```bash
lit-agg profiles
```

## arXiv retrieval

The arXiv integration lives in `src/lit_agg/sources/arxiv_source.py`.

`ArxivSource` implements the `PaperSource` protocol from `sources/base.py`:

```python
fetch_recent(categories, max_results)
fetch_recent_window(categories, start, end, max_results)
search(query, max_results)
```

### Search query mode

For a free-form query:

```python
arxiv.Search(
    query=query,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.Relevance,
)
```

### Recent category mode

For categories:

```python
cat_query = " OR ".join(f"cat:{c}" for c in categories)
```

and sorted by submitted date descending.

### Digest window mode

Digest mode builds a submitted-date query:

```text
(cat:hep-th OR cat:cond-mat.str-el) AND submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
```

### arXiv date-window nuance

arxiv.org category pages show announcement/listing dates, but the API exposes submitted/published timestamps. Papers shown on arxiv.org under a Friday listing may have API timestamps from Thursday UTC.

To avoid missing recent papers, day/week digest windows use a one-day submitted-date lookback. For example:

```bash
lit-agg digest --since 1d
```

may print:

```text
Requested window: 1d
arXiv submittedDate query: 2026-06-04 to 2026-06-06
```

Hour windows such as `--since 24h` remain exact rolling windows.

### arXiv request behavior

`ArxivSource` creates a conservative `arxiv.Client`:

- page size is capped at `MAX_ARXIV_PAGE_SIZE`
- requests use a timeout
- timeouts are normalized into retryable connection errors
- `429` and `503` errors are converted into user-friendly `SourceError`s

This avoids the earlier issue where `--max-papers 10` still requested 100 results due to the arXiv library's default page size.

## Data models

Models live in `src/lit_agg/models.py`.

### `Paper`

Source-agnostic paper metadata:

```python
source: str
source_id: str
title: str
authors: list[str]
abstract: str
published: datetime
url: str
pdf_url: str | None
categories: list[str]
```

### `PaperRelevance`

Used during digest screening:

```python
source_id: str
relevance_score: float
relevance_reason: str
```

### `PaperSummary`

Generated by the summarizer:

```python
source_id: str
summary: str
key_contribution: str
```

### `RankedPaper`

Final display object:

```python
paper: Paper
summary: PaperSummary
relevance_score: float
relevance_reason: str
```

## OpenAI-compatible client

Client setup lives in `src/lit_agg/openai/client.py`.

Base URL priority:

1. `openai_base_url` in config
2. `LIT_AGG_OPENAI_BASE_URL`
3. `OPENAI_BASE_URL`
4. OpenAI SDK default endpoint

API key priority:

1. explicit `--api-key`
2. `api_key_command` from config
3. environment variables

If a custom base URL is configured, the client accepts internal proxy-style key environment variables such as `PI_PROXY_API_KEY`. If no custom base URL is configured, it only uses `OPENAI_API_KEY` to avoid accidentally sending an internal proxy token to public OpenAI.

## LLM stages

All LLM stages use OpenAI structured outputs with JSON schemas.

### Screening

File: `src/lit_agg/openai/screener.py`

Input:

- profile description
- batch of candidate titles/abstracts/categories/dates

Output:

```json
{
  "screenings": [
    {
      "source_id": "...",
      "relevance_score": 7.5,
      "relevance_reason": "..."
    }
  ]
}
```

Screening is intentionally lightweight. It avoids summarizing hundreds of papers.

### Summarization

File: `src/lit_agg/openai/summarizer.py`

Input:

- batch of paper titles and abstracts

Output:

```json
{
  "summaries": [
    {
      "source_id": "...",
      "summary": "...",
      "key_contribution": "..."
    }
  ]
}
```

The prompt explicitly tells the model to only use information supported by the title and abstract.

### Ranking

File: `src/lit_agg/openai/ranker.py`

Input:

- papers
- generated summaries
- either a free-form query or interest profile description

Output:

```json
{
  "rankings": [
    {
      "source_id": "...",
      "relevance_score": 8.0,
      "relevance_reason": "..."
    }
  ]
}
```

Ranked papers are sorted descending by `relevance_score`.

## Display

Terminal rendering lives in `src/lit_agg/display.py`.

It uses Rich panels and includes:

- rank
- relevance score
- title
- authors
- published date
- categories
- summary
- key contribution
- relevance reason
- arXiv URL

Score colors are chosen by `_score_color()`.

## Typical digest execution trace

For:

```bash
lit-agg digest --profile statistics --since 1w --top 10
```

The flow is:

```text
cli.app()
  → command_app
    → digest()
      → load_config()
      → resolve_profile("statistics")
      → _parse_since("1w")
      → get_default_sources()
      → ArxivSource.fetch_recent_window(...)
      → _dedupe_papers(...)
      → get_client(...)
      → screen_papers(...)
      → select top screening results for summary_pool
      → summarize_papers(...)
      → rank_papers(...)
      → display_results(...)
```

## Common extension points

### Add a new profile

Add it under `profiles:` in `config.default.yaml` or in user config:

```yaml
profiles:
  my-topic:
    description: |
      Papers about ...
    default_categories:
      - stat.ME
      - cs.LG
    weekly_top_papers: 10
    weekly_max_candidates: 150
```

### Add a new source

Implement the `PaperSource` protocol:

```python
class MySource:
    @property
    def name(self) -> str: ...
    def fetch_recent(self, categories: list[str], max_results: int) -> list[Paper]: ...
    def fetch_recent_window(self, categories, start, end, max_results) -> list[Paper]: ...
    def search(self, query: str, max_results: int) -> list[Paper]: ...
```

Then return it from `get_default_sources()` in `src/lit_agg/sources/__init__.py`.

### Add export formats

The display step is currently terminal-only. A natural next step is an `export.py` module that can write:

- JSON
- Markdown
- CSV

from a list of `RankedPaper` objects.

### Add caching

The most useful cache points would be:

- arXiv API responses
- screening results by `(profile/prompt hash, paper_id, model)`
- summaries by `(paper_id, model, prompt hash)`
- rankings by `(query/profile hash, paper_ids, model)`

A cache would make repeated weekly runs cheaper and faster.

## Known limitations

- arXiv search is not full semantic search across all historical papers.
- Digest mode only considers fetched candidates from selected categories/date windows.
- Config profiles are not currently deep-merged with built-in profiles.
- Screening/ranking depends on title and abstract quality.
- Final scores are model judgments, not calibrated probabilities.
