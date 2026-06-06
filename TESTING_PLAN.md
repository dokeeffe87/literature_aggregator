# Testing and Evaluation Plan

This document captures the proposed process for testing `lit-agg`, both as working software and as a useful LLM-powered literature recommendation tool.

The goals are to answer four separate questions:

1. **Does the code run?**
2. **Does it retrieve the right papers?**
3. **Are the generated summaries accurate and useful?**
4. **Are the rankings/recommendations high quality?**

The plan is intentionally beginner-friendly for someone with limited prior experience evaluating LLM-generated summaries or recommendations.

---

## Phase 1: Basic local smoke test

Goal: confirm that the package installs and the CLI can run end-to-end.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
lit-agg --help
```

Then run a small live test:

```bash
PI_PROXY_API_KEY=... lit-agg "mechanistic interpretability" --max-papers 3 --verbose
```

Expected outcome:

- arXiv fetch succeeds
- OpenAI-compatible client initializes against the configured endpoint
- summaries are generated
- rankings are generated
- terminal output displays cleanly
- command exits without crashing

This phase only verifies the plumbing. It does **not** establish that the outputs are high quality.

---

## Phase 2: Add non-LLM unit tests

Goal: test deterministic code without calling arXiv or OpenAI.

Suggested test areas:

### Config loading

Target file: `src/lit_agg/config.py`

Test that:

- defaults load correctly
- custom YAML config overrides defaults
- unknown config keys are ignored
- missing config falls back safely

### arXiv result conversion

Target file: `src/lit_agg/sources/arxiv_source.py`

Test that an `arxiv.Result`-like object converts correctly into the internal `Paper` model.

Check fields such as:

- `source`
- `source_id`
- `title`
- `authors`
- `abstract`
- `published`
- `url`
- `pdf_url`
- `categories`

### Display formatting

Target file: `src/lit_agg/display.py`

Test helper behavior such as:

- score-to-color thresholds
- author truncation
- formatting does not fail on long titles or many authors

### Summary and ranking parsing

Target files:

- `src/lit_agg/claude/summarizer.py`
- `src/lit_agg/claude/ranker.py`

Mock OpenAI responses and test that:

- tool-call outputs are parsed correctly
- malformed or missing tool calls are handled sensibly
- rankings are sorted descending by score
- every `RankedPaper` maps back to an existing `Paper` and `PaperSummary`

These tests should be part of normal CI/local test runs and should not require API credentials.

---

## Phase 3: Add small live integration tests

Goal: confirm that real arXiv and OpenAI APIs work together.

Use very small runs to limit cost:

```bash
lit-agg "transformer circuits" --max-papers 3 --verbose
lit-agg --categories cs.CL --max-papers 3 --verbose
```

Check that:

- API key and base URL resolution work
- arXiv returns papers
- OpenAI-generated summaries map back to the correct paper IDs
- every fetched paper receives a summary
- every summarized paper receives a ranking
- output contains valid paper URLs

These tests should likely be optional or marked as live tests, for example:

```bash
pytest -m live
```

Normal test runs should not spend API money.

---

## Phase 4: Create a representative human evaluation set

Goal: evaluate whether the tool is useful for realistic use cases.

Pick 5–10 representative queries that reflect actual intended usage.

Example query set:

```text
mechanistic interpretability for transformers
AI agents for software engineering
LLM evaluation and benchmarks
efficient fine-tuning methods
retrieval augmented generation
AI for nuclear engineering
Bayesian methods for scientific discovery
```

For each query, run:

```bash
lit-agg "<query>" --max-papers 10
```

Save the output for later review.

A useful future improvement would be to add JSON export, e.g.:

```bash
lit-agg "<query>" --max-papers 10 --output results/<query>.json
```

This would make evaluation and regression testing much easier than relying only on terminal output.

---

## Phase 5: Evaluate summary quality

Goal: determine whether generated summaries are faithful, clear, and useful.

For each generated summary, compare it against the paper title and abstract. Rate each dimension from **1 to 5**.

| Dimension | Question |
|---|---|
| Accuracy | Does the summary accurately reflect the abstract? |
| Coverage | Does it capture the main problem, method, and result? |
| Clarity | Is it understandable to a technically literate reader? |
| Usefulness | Does it help decide whether to read the paper? |

Also rate hallucination risk separately:

| Score | Meaning |
|---:|---|
| 0 | No obvious hallucination |
| 1 | Minor unsupported claim |
| 2 | Major unsupported claim |

A good summary should usually score:

- Accuracy: 4–5
- Coverage: 4–5
- Clarity: 4–5
- Usefulness: 4–5
- Hallucination risk: 0

Red flags:

- mentions results not present in the abstract
- overstates significance
- says “state-of-the-art” without support
- invents benchmarks, datasets, or performance numbers
- misses the paper’s actual contribution
- uses vague boilerplate instead of paper-specific content

---

## Phase 6: Evaluate ranking/recommendation quality

Goal: determine whether the tool puts the most useful papers near the top.

For each returned paper, rate the following from **1 to 5**.

| Dimension | Question |
|---|---|
| Query relevance | Is this paper actually related to the query? |
| Importance | Does it seem like a meaningful contribution? |
| Novelty/usefulness | Would I personally consider reading it? |
| Explanation quality | Does the relevance reason justify the score? |

Also evaluate each ranked list as a whole.

| List-level check | Desired behavior |
|---|---|
| Top-3 precision | Most top 3 papers should be clearly relevant |
| Top-5 usefulness | Most top 5 papers should be worth opening |
| Score calibration | Scores of 9–10 should be reserved for truly strong matches |
| Diversity | Top papers should not all be near-duplicates |
| Obvious misses | Highly relevant papers should not be buried below weak ones |
| Relevance reasons | Reasons should cite concrete matches to the query |

A simple beginner-friendly metric:

```text
For each query, how many of the top 5 would I actually open?
```

Rough interpretation:

- 0–2: ranking likely needs substantial improvement
- 3: minimally useful
- 4–5: strong practical utility

---

## Phase 7: Compare against a baseline

Goal: determine whether LLM ranking improves over raw arXiv order.

For each evaluation query:

1. Inspect the raw arXiv top 10 results.
2. Inspect the `lit-agg` ranked top 10 results.
3. Decide which ordering gives a better top 5.

Track results in a table:

| Query | arXiv better | lit-agg better | Tie | Notes |
|---|---:|---:|---:|---|
| mechanistic interpretability |  |  |  |  |
| RAG evaluation |  |  |  |  |
| efficient fine-tuning |  |  |  |  |

No complex metric is required initially. The key question is whether the ranked output saves time compared with the default arXiv ordering.

---

## Phase 8: Add golden-query regression tests

Goal: preserve good behavior once we find it.

Create a small set of fixed “golden queries” with behavioral expectations.

Example:

```text
Query: mechanistic interpretability for transformers
Expected behavior:
- top results should be about interpretability, circuits, probing, attribution, or representation analysis
- papers only about generic LLM scaling should score lower
- summaries should not invent benchmark results
```

These tests should avoid exact expected rankings because LLM output may vary. Prefer behavioral assertions such as:

- all scores are between 0 and 10
- top-ranked relevance reasons mention the query topic
- no duplicate papers are returned
- every summary has a matching source ID
- summaries are non-empty
- relevance reasons are non-empty
- summaries are concise
- output does not contain obvious refusal or uncertainty boilerplate

---

## Phase 9: Optional LLM-as-judge evaluation

Goal: use another LLM to help evaluate many outputs quickly.

Important caveat: LLM-as-judge should assist human review, not replace it.

A judge prompt could evaluate each result using:

- query
- paper title
- paper abstract
- generated summary
- relevance score
- relevance reason

The judge could return structured JSON like:

```json
{
  "summary_accuracy": 4,
  "summary_coverage": 4,
  "summary_clarity": 5,
  "summary_usefulness": 4,
  "hallucination_risk": 0,
  "ranking_relevance": 5,
  "score_calibration": 4,
  "comments": "The summary is faithful to the abstract and the relevance score is justified."
}
```

Use this mainly for:

- regression testing prompt changes
- identifying likely bad summaries
- comparing model or prompt variants

Human spot-checking should still be used, especially for high-impact conclusions.

---

## Phase 10: Recommended repo improvements before serious quality testing

The current repo can run manually, but a few additions would make testing much easier.

### 1. Add a pytest suite

Add unit tests for:

- config loading
- paper model validation
- arXiv conversion
- OpenAI response parsing
- ranking sort order
- display helpers

### 2. Add a fetch-only mode

Useful for testing arXiv retrieval independently from OpenAI.

Possible command:

```bash
lit-agg fetch "mechanistic interpretability" --max-papers 10
```

### 3. Add JSON export

Useful for saving outputs and evaluating them later.

Possible command:

```bash
lit-agg "RAG evaluation" --max-papers 10 --output results/rag_eval.json
```

### 4. Add local caching

Cache:

- arXiv responses
- OpenAI summaries
- OpenAI rankings

This avoids repeated API spend while testing the same queries.

### 5. Add evaluation scripts

Possible future command:

```bash
lit-agg eval results/*.json
```

This could aggregate human or LLM-judge ratings across saved result files.

### 6. Improve error handling

Handle cases where:

- OpenAI does not return the expected structured response
- some papers are missing summaries
- arXiv returns duplicate papers
- API calls fail transiently
- malformed tool output is returned

---

## Recommended first implementation/testing sequence

1. Add basic `pytest` tests for deterministic logic.
2. Add JSON export so results can be saved.
3. Run 5 live evaluation queries with `--max-papers 10`.
4. Manually rate summaries and rankings using the rubrics above.
5. Identify common failure modes.
6. Adjust prompts/model/settings.
7. Re-run the same queries and compare against prior outputs.
8. Add golden-query regression checks once behavior is acceptable.

The key principle is to avoid judging the tool from one anecdotal run. Use a small, repeatable evaluation set and track whether changes actually improve usefulness.
