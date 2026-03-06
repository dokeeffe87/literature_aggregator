"""Batch paper summarization using Claude with tool use and prompt caching."""

import json

import anthropic

from lit_agg.models import Paper, PaperSummary

SYSTEM_PROMPT = """You are an expert research paper analyst. Given a batch of academic papers \
(title + abstract), produce a concise summary and identify the key contribution for each paper.

For each paper, provide:
- summary: A 2-3 sentence summary capturing the main idea, method, and results.
- key_contribution: A single sentence describing the most important contribution.

You MUST call the record_summaries tool with your results."""

RECORD_SUMMARIES_TOOL: anthropic.types.ToolParam = {
    "name": "record_summaries",
    "description": "Record summaries for a batch of papers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
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


def summarize_papers(
    client: anthropic.Anthropic,
    papers: list[Paper],
    model: str,
    batch_size: int = 10,
    verbose: bool = False,
) -> list[PaperSummary]:
    """Summarize papers in batches using Claude tool use with prompt caching."""
    all_summaries: list[PaperSummary] = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        user_content = _format_papers_for_prompt(batch)

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[RECORD_SUMMARIES_TOOL],
            tool_choice={"type": "tool", "name": "record_summaries"},
            messages=[{"role": "user", "content": user_content}],
        )

        if verbose:
            usage = response.usage
            print(
                f"  [Summarizer batch {i // batch_size + 1}] "
                f"input={usage.input_tokens} "
                f"output={usage.output_tokens} "
                f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
                f"cache_create={getattr(usage, 'cache_creation_input_tokens', 0)}"
            )

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_summaries":
                raw = block.input
                if isinstance(raw, str):
                    raw = json.loads(raw)
                for entry in raw["summaries"]:
                    all_summaries.append(PaperSummary(**entry))

    return all_summaries
