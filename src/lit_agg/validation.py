"""Validation helpers for exported lit-agg result files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Severity = Literal["error", "warning"]


@dataclass
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    source_id: str | None = None


PROFILE_POSITIVE_TERMS: dict[str, list[str]] = {
    "physics": [
        "holograph",
        "gauge",
        "gravity",
        "disorder",
        "transport",
        "charged",
        "strange metal",
        "quantum field",
        "nuclear",
    ],
    "holography-transport": [
        "holograph",
        "ads",
        "gauge/gravity",
        "transport",
        "conductivity",
        "diffusion",
        "momentum relaxation",
        "disorder",
        "black brane",
        "hydrodynamic",
        "strange metal",
    ],
    "nuclear-engineering": [
        "reactor",
        "radiation",
        "nuclear",
        "fuel",
        "thermal hydraulic",
        "fission",
        "fusion",
        "plasma",
        "neutron",
        "detector",
    ],
    "ai-research-tools": [
        "agent",
        "tool",
        "retrieval",
        "rag",
        "evaluation",
        "benchmark",
        "code generation",
        "scientific reasoning",
        "workflow",
        "language model",
    ],
    "statistics": [
        "bayesian",
        "inference",
        "uncertainty",
        "hierarchical",
        "nonparametric",
        "high-dimensional",
        "estimator",
        "robust",
        "missing data",
        "experimental design",
    ],
    "causal-inference": [
        "causal",
        "treatment effect",
        "potential outcome",
        "instrumental variable",
        "difference-in-differences",
        "regression discontinuity",
        "synthetic control",
        "experiment",
        "policy evaluation",
        "interference",
    ],
}

_REQUIRED_PAPER_FIELDS = {
    "source_id",
    "title",
    "authors",
    "abstract",
    "published",
    "url",
    "categories",
}
_REQUIRED_SUMMARY_FIELDS = {"source_id", "summary", "key_contribution"}


def load_json_export(path: Path) -> dict[str, Any]:
    """Load an exported result file."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Export root must be a JSON object.")
    return data


def _issue(
    issues: list[ValidationIssue],
    severity: Severity,
    code: str,
    message: str,
    source_id: str | None = None,
) -> None:
    issues.append(ValidationIssue(severity, code, message, source_id))


def _result_score(result: dict[str, Any]) -> float | None:
    score = result.get("relevance_score", result.get("ranking_score"))
    if isinstance(score, bool) or score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _text_blob(result: dict[str, Any]) -> str:
    paper = result.get("paper") if isinstance(result.get("paper"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    parts = [
        paper.get("title"),
        paper.get("abstract"),
        result.get("relevance_reason"),
        result.get("ranking_reason"),
        result.get("screening_reason"),
        summary.get("summary"),
        summary.get("key_contribution"),
    ]
    return " ".join(p for p in parts if isinstance(p, str)).lower()


def _validate_result_shape(
    result: Any,
    index: int,
    issues: list[ValidationIssue],
) -> tuple[str | None, float | None]:
    if not isinstance(result, dict):
        _issue(issues, "error", "invalid-result", f"Result #{index} is not an object.")
        return None, None

    paper = result.get("paper")
    summary = result.get("summary")
    if not isinstance(paper, dict):
        _issue(issues, "error", "missing-paper", f"Result #{index} is missing a paper object.")
        return None, _result_score(result)

    source_id = paper.get("source_id")
    source_id = source_id if isinstance(source_id, str) and source_id else None

    missing_paper = sorted(_REQUIRED_PAPER_FIELDS - set(paper))
    if missing_paper:
        _issue(
            issues,
            "error",
            "missing-paper-fields",
            f"Paper is missing required fields: {', '.join(missing_paper)}.",
            source_id,
        )

    for field in ("title", "abstract", "url", "published"):
        value = paper.get(field)
        if not isinstance(value, str) or not value.strip():
            _issue(issues, "error", "empty-paper-field", f"Paper field '{field}' is empty.", source_id)

    if not isinstance(paper.get("authors"), list) or not paper.get("authors"):
        _issue(issues, "warning", "missing-authors", "Paper has no authors listed.", source_id)

    categories = paper.get("categories")
    if not isinstance(categories, list) or not categories:
        _issue(issues, "warning", "missing-categories", "Paper has no categories listed.", source_id)

    url = paper.get("url")
    if isinstance(url, str) and paper.get("source") == "arxiv" and "arxiv.org/abs/" not in url:
        _issue(issues, "warning", "unexpected-arxiv-url", f"Unexpected arXiv URL: {url}", source_id)

    if not isinstance(summary, dict):
        _issue(issues, "error", "missing-summary", "Result is missing a summary object.", source_id)
    else:
        missing_summary = sorted(_REQUIRED_SUMMARY_FIELDS - set(summary))
        if missing_summary:
            _issue(
                issues,
                "error",
                "missing-summary-fields",
                f"Summary is missing required fields: {', '.join(missing_summary)}.",
                source_id,
            )

        summary_source_id = summary.get("source_id")
        if source_id and summary_source_id and summary_source_id != source_id:
            _issue(
                issues,
                "error",
                "summary-paper-id-mismatch",
                f"Summary source_id {summary_source_id!r} does not match paper source_id {source_id!r}.",
                source_id,
            )

        summary_text = summary.get("summary")
        key_contribution = summary.get("key_contribution")
        if not isinstance(summary_text, str) or not summary_text.strip():
            _issue(issues, "error", "empty-summary", "Summary text is empty.", source_id)
        elif len(summary_text.split()) < 12:
            _issue(issues, "warning", "short-summary", "Summary is unusually short.", source_id)
        elif len(summary_text.split()) > 180:
            _issue(issues, "warning", "long-summary", "Summary is unusually long.", source_id)

        if not isinstance(key_contribution, str) or not key_contribution.strip():
            _issue(issues, "error", "empty-key-contribution", "Key contribution is empty.", source_id)
        elif len(key_contribution.split()) < 5:
            _issue(issues, "warning", "short-key-contribution", "Key contribution is unusually short.", source_id)

    score = _result_score(result)
    if score is None:
        _issue(issues, "error", "invalid-score", "Relevance score is missing or invalid.", source_id)
    elif score < 0 or score > 10:
        _issue(issues, "error", "score-out-of-range", f"Relevance score {score} is outside [0, 10].", source_id)

    reason = result.get("relevance_reason", result.get("ranking_reason"))
    if not isinstance(reason, str) or not reason.strip():
        _issue(issues, "error", "empty-relevance-reason", "Relevance reason is empty.", source_id)
    elif len(reason.split()) < 6:
        _issue(issues, "warning", "short-relevance-reason", "Relevance reason is unusually short.", source_id)

    return source_id, score


def validate_export(data: dict[str, Any]) -> list[ValidationIssue]:
    """Validate a lit-agg JSON export and return errors/warnings."""
    issues: list[ValidationIssue] = []

    if data.get("schema_version") != 1:
        _issue(
            issues,
            "warning",
            "unknown-schema-version",
            f"Expected schema_version 1, got {data.get('schema_version')!r}.",
        )

    results = data.get("results")
    if not isinstance(results, list):
        _issue(issues, "error", "missing-results", "Export is missing a results list.")
        return issues
    if not results:
        _issue(issues, "error", "empty-results", "Export contains no results.")
        return issues

    seen_ids: set[str] = set()
    previous_score: float | None = None
    valid_scores: list[float] = []
    run = data.get("run") if isinstance(data.get("run"), dict) else {}
    requested_categories = set(run.get("categories") or [])
    mode = data.get("mode") or run.get("mode")

    for index, result in enumerate(results, 1):
        source_id, score = _validate_result_shape(result, index, issues)
        if source_id:
            if source_id in seen_ids:
                _issue(issues, "error", "duplicate-paper-id", "Duplicate paper in results.", source_id)
            seen_ids.add(source_id)

        if score is not None:
            valid_scores.append(score)
            if previous_score is not None and score > previous_score:
                _issue(
                    issues,
                    "error",
                    "not-sorted-descending",
                    "Results are not sorted by descending relevance score.",
                    source_id,
                )
            previous_score = score

        if isinstance(result, dict):
            paper = result.get("paper") if isinstance(result.get("paper"), dict) else {}
            paper_categories = set(paper.get("categories") or [])
            if mode == "digest" and requested_categories and paper_categories:
                if requested_categories.isdisjoint(paper_categories):
                    _issue(
                        issues,
                        "warning",
                        "category-mismatch",
                        "Paper categories do not intersect requested digest categories.",
                        source_id,
                    )

    if valid_scores and max(valid_scores[: min(5, len(valid_scores))]) < 5:
        _issue(
            issues,
            "warning",
            "low-top-scores",
            "All top results have relevance scores below 5.",
        )

    profile = run.get("profile")
    terms = PROFILE_POSITIVE_TERMS.get(profile) if isinstance(profile, str) else None
    if terms:
        top_results = [r for r in results[: min(5, len(results))] if isinstance(r, dict)]
        if top_results and not any(
            any(term in _text_blob(result) for term in terms) for result in top_results
        ):
            _issue(
                issues,
                "warning",
                "weak-profile-term-match",
                f"Top results do not contain obvious positive terms for profile '{profile}'.",
            )

    return issues
