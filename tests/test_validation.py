from datetime import UTC, datetime
import unittest

from lit_agg.export import build_run_export
from lit_agg.models import Paper, PaperRelevance, PaperSummary, RankedPaper
from lit_agg.validation import validate_export


def _ranked_paper(source_id: str, score: float, title: str = "Causal treatment effects") -> RankedPaper:
    paper = Paper(
        source="arxiv",
        source_id=source_id,
        title=title,
        authors=["A. Author"],
        abstract="We estimate causal treatment effects in online experiments using robust estimators.",
        published=datetime.now(UTC),
        url=source_id,
        categories=["stat.ME"],
    )
    summary = PaperSummary(
        source_id=source_id,
        summary="This paper studies causal treatment effect estimation in marketplace experiments using robust statistical estimators.",
        key_contribution="It provides practical estimators for causal marketplace experiments.",
    )
    return RankedPaper(
        paper=paper,
        summary=summary,
        relevance_score=score,
        relevance_reason="Directly relevant to causal inference and experiments.",
    )


class ValidationTests(unittest.TestCase):
    def test_valid_export_passes(self) -> None:
        ranked = _ranked_paper("http://arxiv.org/abs/1234.5678v1", 8.0)
        screening = PaperRelevance(
            source_id=ranked.paper.source_id,
            relevance_score=8.5,
            relevance_reason="Matches treatment effects and experiments.",
        )
        export = build_run_export(
            mode="digest",
            ranked_papers=[ranked],
            screening_by_source_id={ranked.paper.source_id: screening},
            run={"mode": "digest", "profile": "causal-inference", "categories": ["stat.ME"]},
            counts={"candidates": 1, "screened": 1, "summarized": 1, "ranked": 1, "displayed": 1},
        )

        self.assertEqual(validate_export(export), [])

    def test_score_out_of_range_is_error(self) -> None:
        ranked = _ranked_paper("http://arxiv.org/abs/1234.5678v1", 8.0)
        export = build_run_export(
            mode="search",
            ranked_papers=[ranked],
            run={"mode": "search", "categories": ["stat.ME"]},
            counts={"candidates": 1, "screened": None, "summarized": 1, "ranked": 1, "displayed": 1},
        )
        export["results"][0]["relevance_score"] = 11

        issues = validate_export(export)
        self.assertIn("score-out-of-range", {issue.code for issue in issues})
        self.assertIn("error", {issue.severity for issue in issues})

    def test_unsorted_scores_are_error(self) -> None:
        export = build_run_export(
            mode="search",
            ranked_papers=[
                _ranked_paper("http://arxiv.org/abs/1111.1111v1", 6.0),
                _ranked_paper("http://arxiv.org/abs/2222.2222v1", 8.0),
            ],
            run={"mode": "search", "categories": ["stat.ME"]},
            counts={"candidates": 2, "screened": None, "summarized": 2, "ranked": 2, "displayed": 2},
        )

        issues = validate_export(export)
        self.assertIn("not-sorted-descending", {issue.code for issue in issues})

    def test_summary_id_mismatch_is_error(self) -> None:
        ranked = _ranked_paper("http://arxiv.org/abs/1234.5678v1", 8.0)
        export = build_run_export(
            mode="search",
            ranked_papers=[ranked],
            run={"mode": "search", "categories": ["stat.ME"]},
            counts={"candidates": 1, "screened": None, "summarized": 1, "ranked": 1, "displayed": 1},
        )
        export["results"][0]["summary"]["source_id"] = "other"

        issues = validate_export(export)
        self.assertIn("summary-paper-id-mismatch", {issue.code for issue in issues})


if __name__ == "__main__":
    unittest.main()
