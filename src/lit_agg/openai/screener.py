"""Lightweight paper relevance screening using OpenAI structured outputs."""

import json
from typing import Any

from openai import OpenAI

from lit_agg.models import Paper, PaperRelevance

SYSTEM_PROMPT = """You are an expert research assistant triaging academic papers for a personalized literature digest.

Given a user's research interests and a batch of paper titles/abstracts, score how relevant each paper is to the user's interests.

Score each paper from 0-10:
- 9-10: Directly relevant and likely worth reading immediately
- 7-8: Strongly relevant
- 5-6: Moderately relevant or useful background
- 3-4: Tangentially related
- 0-2: Not relevant

Be conservative. Base your judgment only on the title, abstract, categories, and publication date. Return a screening entry for every paper listed."""

SCREENING_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_screenings",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "screenings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "The source_id of the paper being screened.",
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
            "required": ["screenings"],
        },
    },
}


def _format_papers_for_screening(papers: list[Paper]) -> str:
    parts = []
    for i, p in enumerate(papers, 1):
        published = p.published.date().isoformat()
        parts.append(
            f"--- Paper {i} ---\n"
            f"Source ID: {p.source_id}\n"
            f"Title: {p.title}\n"
            f"Published: {published}\n"
            f"Categories: {', '.join(p.categories)}\n"
            f"Abstract: {p.abstract}\n"
        )
    return "\n".join(parts)


def _usage_value(usage: Any, field: str) -> int:
    return int(getattr(usage, field, 0) or 0) if usage is not None else 0


def screen_papers(
    client: OpenAI,
    papers: list[Paper],
    interests: str,
    model: str,
    batch_size: int = 20,
    verbose: bool = False,
) -> list[PaperRelevance]:
    """Score candidate papers against a free-form interest profile."""
    screenings: list[PaperRelevance] = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        user_content = (
            "User research interests:\n"
            f"{interests.strip()}\n\n"
            "Candidate papers:\n"
            f"{_format_papers_for_screening(batch)}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=SCREENING_RESPONSE_FORMAT,
            temperature=0.1,
            max_tokens=4096,
        )

        if verbose:
            usage = response.usage
            print(
                f"  [Screening batch {i // batch_size + 1}] "
                f"input={_usage_value(usage, 'prompt_tokens')} "
                f"output={_usage_value(usage, 'completion_tokens')} "
                f"total={_usage_value(usage, 'total_tokens')}"
            )

        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"OpenAI refused the screening request: {refusal}")

        content = message.content
        if not content:
            raise ValueError("OpenAI returned an empty screening response")

        raw = json.loads(content)
        for entry in raw["screenings"]:
            screenings.append(PaperRelevance(**entry))

    screenings.sort(key=lambda r: r.relevance_score, reverse=True)
    return screenings
