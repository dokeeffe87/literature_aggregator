"""Rich terminal display for ranked papers."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from lit_agg.models import RankedPaper

console = Console()


def _score_color(score: float) -> str:
    if score >= 8:
        return "bold green"
    if score >= 6:
        return "green"
    if score >= 4:
        return "yellow"
    if score >= 2:
        return "red"
    return "dim red"


def _truncate_authors(authors: list[str], max_shown: int = 3) -> str:
    if len(authors) <= max_shown:
        return ", ".join(authors)
    return ", ".join(authors[:max_shown]) + f" (+{len(authors) - max_shown} more)"


def display_results(
    ranked_papers: list[RankedPaper],
    query: str | None = None,
) -> None:
    """Display ranked papers with Rich formatting."""
    if query:
        header = f'Results for: "{query}"'
    else:
        header = "Results ranked by general interest"

    console.print()
    console.print(
        Panel(
            f"[bold]{header}[/bold]\n"
            f"[dim]{len(ranked_papers)} papers ranked[/dim]",
            style="blue",
        )
    )
    console.print()

    for rank, rp in enumerate(ranked_papers, 1):
        score = rp.relevance_score
        color = _score_color(score)
        authors = _truncate_authors(rp.paper.authors)

        score_text = Text(f" {score:.1f}/10 ", style=f"{color} reverse")

        content = Text()
        content.append(f"#{rank} ", style="bold")
        content.append_text(score_text)
        content.append(f" {rp.paper.title}\n", style="bold")
        content.append(f"   {authors}\n", style="dim")
        content.append(f"\n   {rp.summary.summary}\n")
        content.append(f"\n   Key: ", style="bold")
        content.append(f"{rp.summary.key_contribution}\n")
        content.append(f"\n   Relevance: ", style="bold")
        content.append(f"{rp.relevance_reason}\n")
        content.append(f"\n   {rp.paper.url}", style="dim underline")

        console.print(Panel(content, border_style=color))
        console.print()
