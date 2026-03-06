"""Paper source registry."""

from lit_agg.sources.arxiv_source import ArxivSource
from lit_agg.sources.base import PaperSource


def get_default_sources() -> list[PaperSource]:
    """Return all available paper sources."""
    return [ArxivSource()]
