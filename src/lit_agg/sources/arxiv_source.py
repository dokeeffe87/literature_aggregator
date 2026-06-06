"""ArXiv paper source using the arxiv PyPI package."""

import time
from datetime import UTC, datetime

import arxiv
import requests

from lit_agg.models import Paper
from lit_agg.sources.base import SourceError

# Keep pages small so a small --max-papers request does not ask arXiv for the
# library default of 100 results. Smaller pages also make transient 503/429
# responses less likely when arXiv is under load.
MAX_ARXIV_PAGE_SIZE = 25
ARXIV_DELAY_SECONDS = 4.0
ARXIV_NUM_RETRIES = 3
ARXIV_REQUEST_TIMEOUT_SECONDS = 10.0


def _format_arxiv_datetime(dt: datetime) -> str:
    """Format a datetime for arXiv submittedDate range queries."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y%m%d%H%M")


class _TimeoutSession(requests.Session):
    def request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("timeout", ARXIV_REQUEST_TIMEOUT_SECONDS)
        try:
            return super().request(method, url, **kwargs)
        except requests.Timeout as e:
            # arxiv.Client retries ConnectionError but not Timeout, so normalize
            # timeouts into the retryable exception family and keep the retry
            # cadence polite even though arxiv.Client did not record a response.
            time.sleep(ARXIV_DELAY_SECONDS)
            raise requests.ConnectionError(str(e)) from e


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

    def _client(self, max_results: int) -> arxiv.Client:
        """Create a conservative arXiv client for the requested result count."""
        page_size = max(1, min(max_results, MAX_ARXIV_PAGE_SIZE))
        client = arxiv.Client(
            page_size=page_size,
            delay_seconds=ARXIV_DELAY_SECONDS,
            num_retries=ARXIV_NUM_RETRIES,
        )
        client._session = _TimeoutSession()
        return client

    def _results(self, search: arxiv.Search, max_results: int) -> list[Paper]:
        client = self._client(max_results)
        try:
            return [self._convert(r) for r in client.results(search)]
        except arxiv.HTTPError as e:
            if e.status in {429, 503}:
                detail = (
                    f"arXiv API returned HTTP {e.status} after retries. "
                    "This is usually temporary rate limiting or service load; "
                    "wait a few minutes and retry."
                )
            else:
                detail = f"arXiv API returned HTTP {e.status}."
            raise SourceError(detail) from e
        except arxiv.ArxivError as e:
            raise SourceError(f"arXiv request failed: {e}") from e
        except requests.RequestException as e:
            raise SourceError(f"arXiv request failed before receiving a response: {e}") from e

    def fetch_recent(self, categories: list[str], max_results: int) -> list[Paper]:
        """Fetch recent papers from given arxiv categories."""
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        search = arxiv.Search(
            query=cat_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        return self._results(search, max_results)

    def fetch_recent_window(
        self,
        categories: list[str],
        start: datetime,
        end: datetime,
        max_results: int,
    ) -> list[Paper]:
        """Fetch papers submitted in a time window from given arXiv categories."""
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        start_s = _format_arxiv_datetime(start)
        end_s = _format_arxiv_datetime(end)
        query = f"({cat_query}) AND submittedDate:[{start_s} TO {end_s}]"
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        return self._results(search, max_results)

    def search(self, query: str, max_results: int) -> list[Paper]:
        """Search arxiv for papers matching a query."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return self._results(search, max_results)
