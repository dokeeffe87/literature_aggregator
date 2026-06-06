"""Paper ranking using OpenAI structured outputs."""

import json
from typing import Any

from openai import OpenAI

from lit_agg.models import Paper, PaperSummary, RankedPaper

SYSTEM_PROMPT_TEMPLATE = """You are an expert research paper ranker. You will be given a list of \
papers with their summaries. Rank them by relevance and importance.

{query_instruction}

Score each paper from 0-10:
- 9-10: Directly relevant, high-impact, must-read
- 7-8: Strongly relevant, notable contribution
- 5-6: Moderately relevant or interesting
- 3-4: Tangentially related
- 0-2: Not relevant

Use the full score range, but be conservative: a paper should only receive 9-10 if it is an unusually strong match. Base your judgments only on the provided title, categories, summary, and key contribution.

Return a ranking for every paper listed."""

RANKING_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_rankings",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rankings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "The source_id of the paper.",
                            },
                            "relevance_score": {
                                "type": "number",
                                "description": "Relevance score from 0 to 10.",
                            },
                            "relevance_reason": {
                                "type": "string",
                                "description": "Brief explanation of the relevance score.",
                            },
                        },
                        "required": ["source_id", "relevance_score", "relevance_reason"],
                    },
                }
            },
            "required": ["rankings"],
        },
    },
}


def _build_system_prompt(query: str | None) -> str:
    if query:
        instruction = f'Rank by relevance to this research interest: "{query}"'
    else:
        instruction = (
            "No specific query provided. Rank by general scientific interest, "
            "novelty, and potential impact."
        )
    return SYSTEM_PROMPT_TEMPLATE.format(query_instruction=instruction)


def _format_papers_with_summaries(
    papers: list[Paper], summaries: list[PaperSummary]
) -> str:
    summary_map = {s.source_id: s for s in summaries}
    parts = []
    for i, p in enumerate(papers, 1):
        s = summary_map.get(p.source_id)
        summary_text = s.summary if s else "No summary available."
        key_contrib = s.key_contribution if s else "N/A"
        parts.append(
            f"--- Paper {i} ---\n"
            f"Source ID: {p.source_id}\n"
            f"Title: {p.title}\n"
            f"Summary: {summary_text}\n"
            f"Key Contribution: {key_contrib}\n"
            f"Categories: {', '.join(p.categories)}\n"
        )
    return "\n".join(parts)


def _usage_value(usage: Any, field: str) -> int:
    return int(getattr(usage, field, 0) or 0) if usage is not None else 0


def rank_papers(
    client: OpenAI,
    papers: list[Paper],
    summaries: list[PaperSummary],
    model: str,
    query: str | None = None,
    verbose: bool = False,
) -> list[RankedPaper]:
    """Rank papers using OpenAI structured outputs."""
    system_prompt = _build_system_prompt(query)
    user_content = _format_papers_with_summaries(papers, summaries)
    paper_map = {p.source_id: p for p in papers}
    summary_map = {s.source_id: s for s in summaries}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=RANKING_RESPONSE_FORMAT,
        temperature=0.1,
        max_tokens=4096,
    )

    if verbose:
        usage = response.usage
        print(
            f"  [Ranker] "
            f"input={_usage_value(usage, 'prompt_tokens')} "
            f"output={_usage_value(usage, 'completion_tokens')} "
            f"total={_usage_value(usage, 'total_tokens')}"
        )

    message = response.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"OpenAI refused the ranking request: {refusal}")

    content = message.content
    if not content:
        raise ValueError("OpenAI returned an empty ranking response")

    raw = json.loads(content)
    ranked: list[RankedPaper] = []
    for entry in raw["rankings"]:
        source_id = entry["source_id"]
        paper = paper_map.get(source_id)
        summary = summary_map.get(source_id)
        if paper and summary:
            ranked.append(
                RankedPaper(
                    paper=paper,
                    summary=summary,
                    relevance_score=entry["relevance_score"],
                    relevance_reason=entry["relevance_reason"],
                )
            )

    ranked.sort(key=lambda r: r.relevance_score, reverse=True)
    return ranked
