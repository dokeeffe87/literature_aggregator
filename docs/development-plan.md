# Development plan

This document preserves the next-phase roadmap for `lit-agg` so design ideas do not get lost. It assumes the current Phase 1 digest implementation is in place.

## Current state: Phase 1 complete

Implemented capabilities:

- default arXiv search flow:
  - `lit-agg "query" --max-papers 10`
  - `lit-agg --categories cs.AI,cs.LG --max-papers 10`
- personalized digest flow:
  - `lit-agg digest --profile physics --since 7d --top 10`
- interest profiles:
  - built-in profiles for physics, holography/transport, nuclear engineering, AI research tools, statistics, and causal inference
  - profile listing via `lit-agg profiles`
- arXiv category/date-window retrieval
- digest relevance screening before summarization
- LLM summarization and final ranking
- improved arXiv rate-limit/timeout handling
- basic docs:
  - `docs/how-it-works.md`
  - this development plan

The most important design pattern now in place is:

```text
candidate retrieval → cheap screening → shortlist summarization → final ranking → display
```

That pattern should remain the core of future features.

---

## Phase 2: research-question mode

Goal: support a richer research-question workflow where the user describes the problem they are trying to solve and gets a ranked set of relevant papers, regardless of recency.

Example UX:

```bash
lit-agg question \
  "What are the best holographic models of charge transport in disordered systems?" \
  --categories hep-th,cond-mat.str-el \
  --max-candidates 300 \
  --top 15
```

Alternative UX:

```bash
lit-agg question --file question.md --profile holography-transport --top 15
```

### Why this is different from current search

The current default search mode sends the raw query to arXiv and summarizes/ranks the returned results. That is useful, but it is limited by arXiv's keyword search.

Research-question mode should do more:

1. Read the user's free-form question.
2. Generate multiple arXiv search queries from it.
3. Fetch candidates from all generated queries.
4. Deduplicate candidates.
5. Screen titles/abstracts against the original question, not just the generated keywords.
6. Summarize only the strongest candidates.
7. Final-rank the summaries against the original question.

### Proposed pipeline

```text
question/profile
      ↓
LLM search planner
      ↓
multiple arXiv searches
      ↓
deduped candidate pool
      ↓
LLM abstract screening
      ↓
summary pool
      ↓
LLM summaries
      ↓
LLM final ranking
      ↓
display/export
```

### Proposed new module

```text
src/lit_agg/openai/search_planner.py
```

Potential structured output:

```python
class SearchPlan(BaseModel):
    original_question: str
    arxiv_queries: list[str]
    categories: list[str] | None
    rationale: str
```

Example generated queries for a holography/transport question:

```text
all:"holographic disorder" AND all:"charge transport"
abs:"momentum relaxation" AND abs:"conductivity" AND cat:hep-th
all:"gauge gravity" AND all:"disorder" AND all:"transport"
abs:"black brane" AND abs:"electric conductivity"
```

### CLI options to consider

```bash
lit-agg question "..." \
  --profile holography-transport \
  --categories hep-th,cond-mat.str-el \
  --max-candidates 300 \
  --queries 6 \
  --top 15 \
  --summary-pool 30
```

Possible flags:

- `--profile`: optional interest profile to bias query planning and ranking
- `--categories`: optional category constraints
- `--max-candidates`: total candidate papers across all generated arXiv queries
- `--queries`: number of arXiv search queries to generate
- `--top`: final number of displayed papers
- `--summary-pool`: number of screened papers to summarize
- `--file`: read long question/context from a Markdown/text file
- `--show-plan`: print generated arXiv queries before fetching

### Risks / limitations

- arXiv search is still keyword-based, so all-time semantic recall will be imperfect.
- Query expansion can over-broaden and produce noisy candidates.
- Some important older papers may not be retrieved if the generated search strings miss their terminology.

### Acceptance criteria

- Can answer a broad research question with results that are not merely title keyword matches.
- Prints or optionally exposes the generated arXiv query plan.
- Deduplicates candidates across generated queries.
- Screens against the original user question.
- Summarizes only the best-screened candidates.

---

## Phase 3: export and caching

Goal: make results reusable, comparable, cheaper, and easier to evaluate.

### 3A: JSON and Markdown export

Example UX:

```bash
lit-agg digest --profile statistics --since 1w --top 10 --output results/statistics.md
lit-agg question "..." --output results/question.json
```

Proposed module:

```text
src/lit_agg/export.py
```

Export formats:

- JSON: complete structured data for downstream evaluation
- Markdown: readable digest/report
- CSV: optional table export for spreadsheet review

JSON should include:

```json
{
  "run": {
    "mode": "digest",
    "profile": "statistics",
    "query": null,
    "categories": ["stat.ME", "stat.TH"],
    "since": "1w",
    "models": {
      "screen": "...",
      "summarize": "...",
      "rank": "..."
    },
    "created_at": "..."
  },
  "candidates_count": 150,
  "screened_count": 150,
  "summarized_count": 20,
  "results": [
    {
      "rank": 1,
      "relevance_score": 8.5,
      "relevance_reason": "...",
      "paper": {
        "source_id": "...",
        "title": "...",
        "authors": ["..."],
        "abstract": "...",
        "published": "...",
        "url": "...",
        "categories": ["..."]
      },
      "summary": {
        "summary": "...",
        "key_contribution": "..."
      }
    }
  ]
}
```

Markdown should include:

- run metadata
- ranked paper list
- score and relevance reason
- summary/key contribution
- paper URLs

### 3B: local caching

Goal: avoid repeatedly spending time/tokens on the same papers.

Possible cache directory:

```text
~/.cache/lit-agg/
```

Useful cache points:

1. arXiv API responses
2. screening results
3. summaries
4. final rankings

Suggested cache keys:

```text
arxiv:
  hash(source, query, start, end, max_results, page_size)

screening:
  hash(model, prompt_version, profile_or_question_hash, paper_source_id, paper_updated_or_published)

summary:
  hash(model, prompt_version, paper_source_id, paper_updated_or_published)

ranking:
  hash(model, prompt_version, query_or_profile_hash, ordered_paper_ids, summary_hashes)
```

A simple first version could use SQLite:

```text
~/.cache/lit-agg/cache.sqlite
```

or JSON files grouped by cache type. SQLite will likely be easier once the cache grows.

### CLI options to consider

```bash
--cache / --no-cache
--refresh
--cache-dir ~/.cache/lit-agg
```

### Acceptance criteria

- Re-running the same digest does not re-summarize unchanged papers.
- Cache can be bypassed with `--refresh`.
- Exported JSON can be used for later evaluation and regression testing.

---

## Phase 4: local paper index

Goal: improve all-time retrieval beyond arXiv keyword search by keeping a local searchable metadata/index store.

This is closest to the older `Abstraction` design that used precomputed vectorized paper representations.

### Proposed UX

```bash
lit-agg index build --categories hep-th,cond-mat.str-el --from 2000-01-01
lit-agg index update --categories hep-th,cond-mat.str-el
lit-agg question "..." --use-index --top 15
```

### Index contents

Use SQLite as the primary metadata store:

```text
papers
  arxiv_id
  title
  abstract
  authors
  categories
  primary_category
  published
  updated
  url
  pdf_url

paper_terms or paper_embeddings
  arxiv_id
  representation_type
  vector/blob/json
  model_or_method
  created_at
```

### Retrieval options

Start simple:

1. BM25 or TF-IDF over title + abstract
2. optional semantic embeddings later

Candidate retrieval pipeline with an index:

```text
question/profile
      ↓
local BM25 / TF-IDF / embedding retrieval
      ↓
top 200 local candidates
      ↓
LLM abstract screening
      ↓
summary pool
      ↓
LLM summary + final ranking
```

### Why this matters

The arXiv API does not provide high-quality semantic retrieval over all historical papers. A local index can support:

- better all-time recall
- repeatable evaluation
- faster candidate retrieval
- future offline workflows
- hybrid keyword + semantic search

### Implementation steps

1. Add `src/lit_agg/index/` package.
2. Add SQLite schema and migration/init logic.
3. Add index build/update commands.
4. Store arXiv metadata for chosen categories/date ranges.
5. Add BM25 or TF-IDF retrieval.
6. Wire `lit-agg question --use-index` into the existing screening/summarization/ranking pipeline.
7. Optionally add embeddings after the lexical index works.

### Acceptance criteria

- Can build an index for a category/date range.
- Can update incrementally without duplicating papers.
- Can retrieve candidate papers for a free-form question without hitting arXiv search.
- Can feed local candidates through existing LLM screening/summarization/ranking.

---

## Phase 5: evaluation and tests

Goal: make quality improvements measurable and prevent regressions.

### Unit tests

Add tests for deterministic code:

- config loading
- profile validation
- `--since` parsing/windowing
- arXiv query construction
- paper conversion
- deduplication
- output/export formatting
- structured response parsing

### Golden-query regression tests

Use fixed queries/profiles with behavioral expectations.

Examples:

```text
Profile: holography-transport
Expected behavior:
- top results should mention holography, transport, disorder, black branes, hydrodynamics, or strange metals
- generic GR/cosmology should score lower unless directly tied to transport/holography

Profile: causal-inference
Expected behavior:
- top results should involve treatment effects, identification, experiments, IV, diff-in-diff, causal ML, or policy evaluation
- generic prediction-only ML papers should score lower
```

Avoid exact expected rankings because LLM output can vary. Prefer assertions such as:

- scores are within 0-10
- every ranked paper has a matching summary
- summaries are non-empty
- relevance reasons are non-empty
- no duplicate papers
- top papers contain profile-relevant terms or judge as relevant

### Human evaluation set

Maintain a small set of realistic tasks:

```text
holography and disordered charge transport
causal inference for marketplace experiments
Bayesian hierarchical models for noisy measurement
LLM agents for research workflows
nuclear engineering methods for reactor safety
```

For each, save output JSON/Markdown and manually rate:

- relevance
- summary faithfulness
- usefulness
- diversity
- score calibration

### LLM-as-judge evaluation

Optional after JSON export exists. Use another model to judge:

- summary faithfulness to abstract
- relevance to query/profile
- whether score/reason is justified
- hallucination risk

Human review should remain the source of truth for important decisions.

---

## Phase 6: product polish

Potential improvements after the core workflows are stable:

- Better command structure without manual Typer dispatch, if preserving old UX remains manageable.
- `--dry-run` / `--show-candidates` / `--show-screening` debug modes.
- Cost/token estimates before running large digests.
- Config command helpers:
  - `lit-agg config path`
  - `lit-agg profiles show PROFILE`
  - `lit-agg profiles create PROFILE`
- Better terminal summaries:
  - grouped by category
  - score thresholds
  - collapsed abstracts
- Optional scheduled digest runner.
- Optional Slack/email/Markdown report integration.

---

## Open design questions

1. Should user config deep-merge with built-in profiles?
   - Recommended: yes.
   - User-defined profiles should be added to built-ins, and same-name profiles should override built-ins.

2. Should profile descriptions be free-form only, or structured?
   - Free-form is flexible.
   - Structured fields like `prefer`, `deprioritize`, and `keywords` could improve prompts and future indexing.

3. Should screening and ranking be separate prompts long-term?
   - Current answer: yes.
   - Screening is cheap triage; ranking sees summaries and should be more nuanced.

4. Should embeddings be required?
   - Not initially.
   - BM25/TF-IDF plus LLM reranking may be strong enough for many use cases.

5. What should be cached first?
   - Summaries are likely the highest-value cache because they are paper-specific and reusable across profiles/questions.

---

## Recommended immediate next steps

1. Implement Phase 2 research-question mode using LLM query planning plus arXiv candidate retrieval.
2. Add JSON/Markdown export so results can be saved and evaluated.
3. Add summary caching before doing large repeated runs.
4. Add basic unit tests for config/profile/windowing/deduplication.
5. Revisit local indexing once question mode exposes the practical limits of arXiv keyword search.
