"""Batch paper summarization using OpenAI structured outputs."""

import json
from typing import Any

from openai import OpenAI

from lit_agg.models import Paper, PaperSummary

SYSTEM_PROMPT = """You are an expert research paper analyst. Given a batch of academic papers \
(title + abstract), produce a concise, faithful summary and identify the key contribution for each paper.

For each paper, provide:
- summary: A 2-3 sentence summary capturing the main idea, method, and results.
- key_contribution: A single sentence describing the most important contribution.

Only use information supported by the title and abstract. Do not invent benchmarks, datasets, metrics, or claims that are not present in the abstract."""

SUMMARY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_summaries",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "The source_id of the paper being summarized.",
                            },
                            "summary": {
                                "type": "string",
                                "description": "2-3 sentence summary of the paper.",
                            },
                            "key_contribution": {
                                "type": "string",
                                "description": "One sentence describing the key contribution.",
                            },
                        },
                        "required": ["source_id", "summary", "key_contribution"],
                    },
                }
            },
            "required": ["summaries"],
        },
    },
}


def _format_papers_for_prompt(papers: list[Paper]) -> str:
    parts = []
    for i, p in enumerate(papers, 1):
        parts.append(
            f"--- Paper {i} ---\n"
            f"Source ID: {p.source_id}\n"
            f"Title: {p.title}\n"
            f"Abstract: {p.abstract}\n"
        )
    return "\n".join(parts)


def _usage_value(usage: Any, field: str) -> int:
    return int(getattr(usage, field, 0) or 0) if usage is not None else 0


def summarize_papers(
    client: OpenAI,
    papers: list[Paper],
    model: str,
    batch_size: int = 10,
    verbose: bool = False,
) -> list[PaperSummary]:
    """Summarize papers in batches using OpenAI structured outputs."""
    all_summaries: list[PaperSummary] = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        user_content = _format_papers_for_prompt(batch)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=SUMMARY_RESPONSE_FORMAT,
            temperature=0.2,
            max_tokens=4096,
        )

        if verbose:
            usage = response.usage
            print(
                f"  [Summarizer batch {i // batch_size + 1}] "
                f"input={_usage_value(usage, 'prompt_tokens')} "
                f"output={_usage_value(usage, 'completion_tokens')} "
                f"total={_usage_value(usage, 'total_tokens')}"
            )

        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"OpenAI refused the summary request: {refusal}")

        content = message.content
        if not content:
            raise ValueError("OpenAI returned an empty summary response")

        raw = json.loads(content)
        for entry in raw["summaries"]:
            all_summaries.append(PaperSummary(**entry))

    return all_summaries
