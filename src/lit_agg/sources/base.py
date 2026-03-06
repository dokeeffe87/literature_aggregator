"""Protocol for paper sources."""

from typing import Protocol

from lit_agg.models import Paper


class PaperSource(Protocol):
    @property
    def name(self) -> str: ...

    def fetch_recent(self, categories: list[str], max_results: int) -> list[Paper]: ...

    def search(self, query: str, max_results: int) -> list[Paper]: ...
