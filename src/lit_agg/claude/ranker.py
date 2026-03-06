"""Paper ranking using Claude with extended thinking, tool use, and prompt caching."""

import json

import anthropic

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

You MUST call the record_rankings tool with your rankings for ALL papers."""

RECORD_RANKINGS_TOOL: anthropic.types.ToolParam = {
    "name": "record_rankings",
    "description": "Record relevance rankings for all papers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
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


def rank_papers(
    client: anthropic.Anthropic,
    papers: list[Paper],
    summaries: list[PaperSummary],
    model: str,
    query: str | None = None,
    verbose: bool = False,
) -> list[RankedPaper]:
    """Rank papers using Claude with extended thinking."""
    system_prompt = _build_system_prompt(query)
    user_content = _format_papers_with_summaries(papers, summaries)
    summary_map = {s.source_id: s for s in summaries}

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[RECORD_RANKINGS_TOOL],
        # tool_choice must be auto when extended thinking is enabled
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": user_content}],
    )

    if verbose:
        usage = response.usage
        print(
            f"  [Ranker] "
            f"input={usage.input_tokens} "
            f"output={usage.output_tokens} "
            f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
            f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}"
        )
        for block in response.content:
            if block.type == "thinking":
                print(f"  [Thinking] {block.thinking[:200]}...")

    ranked: list[RankedPaper] = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_rankings":
            raw = block.input
            if isinstance(raw, str):
                raw = json.loads(raw)
            for entry in raw["rankings"]:
                source_id = entry["source_id"]
                paper = next((p for p in papers if p.source_id == source_id), None)
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
