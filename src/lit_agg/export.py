"""JSON export helpers for lit-agg runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lit_agg.models import PaperRelevance, RankedPaper

SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def build_run_export(
    *,
    mode: str,
    ranked_papers: list[RankedPaper],
    run: dict[str, Any],
    counts: dict[str, Any],
    screening_by_source_id: dict[str, PaperRelevance] | None = None,
) -> dict[str, Any]:
    """Build a stable JSON-serializable export for a completed run."""
    screening_by_source_id = screening_by_source_id or {}

    results: list[dict[str, Any]] = []
    for rank, ranked in enumerate(ranked_papers, 1):
        source_id = ranked.paper.source_id
        screening = screening_by_source_id.get(source_id)
        results.append(
            {
                "rank": rank,
                "paper": ranked.paper.model_dump(mode="json"),
                "summary": ranked.summary.model_dump(mode="json"),
                "relevance_score": ranked.relevance_score,
                "relevance_reason": ranked.relevance_reason,
                "ranking_score": ranked.relevance_score,
                "ranking_reason": ranked.relevance_reason,
                "screening_score": screening.relevance_score if screening else None,
                "screening_reason": screening.relevance_reason if screening else None,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "run": _jsonable(run),
        "counts": _jsonable(counts),
        "results": results,
    }


def write_json_export(export: dict[str, Any], path: Path) -> None:
    """Write a run export to a JSON file, creating parent directories."""
    if path.suffix.lower() != ".json":
        raise ValueError("Only JSON export is currently supported; use a .json path.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
        f.write("\n")
