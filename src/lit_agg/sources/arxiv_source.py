"""ArXiv paper source using the arxiv PyPI package."""

import arxiv

from lit_agg.models import Paper


class ArxivSource:
    @property
    def name(self) -> str:
        return "arxiv"

    def _convert(self, result: arxiv.Result) -> Paper:
        return Paper(
            source="arxiv",
            source_id=result.entry_id,
            title=result.title.replace("\n", " ").strip(),
            authors=[a.name for a in result.authors],
            abstract=result.summary.replace("\n", " ").strip(),
            published=result.published,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            categories=result.categories,
        )

    def fetch_recent(self, categories: list[str], max_results: int) -> list[Paper]:
        """Fetch recent papers from given arxiv categories."""
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        search = arxiv.Search(
            query=cat_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        client = arxiv.Client()
        return [self._convert(r) for r in client.results(search)]

    def search(self, query: str, max_results: int) -> list[Paper]:
        """Search arxiv for papers matching a query."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()
        return [self._convert(r) for r in client.results(search)]
