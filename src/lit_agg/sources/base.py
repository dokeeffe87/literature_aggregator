"""Protocol and shared errors for paper sources."""

from datetime import datetime
from typing import Protocol

from lit_agg.models import Paper


class SourceError(RuntimeError):
    """Raised when a paper source cannot fetch results."""


class PaperSource(Protocol):
    @property
    def name(self) -> str: ...

    def fetch_recent(self, categories: list[str], max_results: int) -> list[Paper]: ...

    def fetch_recent_window(
        self,
        categories: list[str],
        start: datetime,
        end: datetime,
        max_results: int,
    ) -> list[Paper]: ...

    def search(self, query: str, max_results: int) -> list[Paper]: ...
