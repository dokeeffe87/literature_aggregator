"""CLI entry point for lit-agg."""

from datetime import UTC, datetime, timedelta
import re
from textwrap import shorten
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console

from lit_agg.config import Config, load_config
from lit_agg.display import display_results
from lit_agg.models import Paper
from lit_agg.openai.client import get_client
from lit_agg.openai.ranker import rank_papers
from lit_agg.openai.screener import screen_papers
from lit_agg.openai.summarizer import summarize_papers
from lit_agg.profiles import ProfileError, resolve_profile
from lit_agg.sources import get_default_sources
from lit_agg.sources.base import SourceError

search_app = typer.Typer(
    name="lit-agg",
    help=(
        "Fetch, summarize, and rank recent research papers using OpenAI. "
        "For personalized category digests, run: lit-agg digest --help. "
        "For configured profiles, run: lit-agg profiles."
    ),
    no_args_is_help=False,
)
command_app = typer.Typer(
    name="lit-agg",
    help="Personalized literature aggregation commands.",
    no_args_is_help=True,
)
console = Console()

_COMMANDS = {"digest", "profiles"}
_SINCE_RE = re.compile(r"^(?P<count>\d+)(?P<unit>h|d|w)$")
# arXiv's website groups papers by announcement date, but the API's
# submittedDate/published timestamp is usually the previous UTC date for those
# announcements. For user-facing day/week digests, query one extra submitted
# day so `--since 1d` includes the papers shown on arxiv.org as yesterday/today.
_ARXIV_ANNOUNCEMENT_LOOKBACK_DAYS = 1


@command_app.callback()
def command_root() -> None:
    """Personalized literature aggregation commands."""


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _start_of_utc_day(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_since(value: str) -> datetime:
    """Parse a digest window start.

    Day/week windows are calendar-day based, not rolling-to-the-second. arXiv's
    API exposes submitted timestamps while arxiv.org shows announcement dates,
    so day/week digest queries include one extra submitted day as a safety
    offset. Hour windows remain exact rolling windows.
    """
    normalized = value.strip().lower()
    now = datetime.now(UTC)

    if normalized in {"week", "weekly"}:
        return _start_of_utc_day(
            now - timedelta(days=7 + _ARXIV_ANNOUNCEMENT_LOOKBACK_DAYS)
        )

    match = _SINCE_RE.match(normalized)
    if match:
        count = int(match.group("count"))
        unit = match.group("unit")
        if unit == "h":
            return now - timedelta(hours=count)
        days = count if unit == "d" else count * 7
        return _start_of_utc_day(
            now - timedelta(days=days + _ARXIV_ANNOUNCEMENT_LOOKBACK_DAYS)
        )

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise typer.BadParameter(
            "Use a relative duration like 7d, 1w, or 24h, or an ISO date like 2026-06-01."
        ) from e

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return _start_of_utc_day(parsed)


def _client_or_exit(config: Config, api_key: str | None):
    try:
        return get_client(config, api_key=api_key)
    except Exception as e:
        console.print(f"[red]Failed to create API client: {e}[/red]")
        console.print(
            "[dim]Set PI_PROXY_API_KEY for Shopify AI Proxy, set OPENAI_API_KEY for direct OpenAI, use --api-key, or configure api_key_command.[/dim]"
        )
        raise typer.Exit(1)


def _dedupe_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    deduped: list[Paper] = []
    for paper in papers:
        if paper.source_id in seen:
            continue
        seen.add(paper.source_id)
        deduped.append(paper)
    return deduped


@search_app.command()
def main(
    query: Annotated[
        Optional[str],
        typer.Argument(help="Natural language research interest (optional)."),
    ] = None,
    categories: Annotated[
        Optional[str],
        typer.Option(
            "--categories",
            "-c",
            help="Comma-separated arxiv categories (e.g. cs.AI,cs.LG).",
        ),
    ] = None,
    max_papers: Annotated[
        Optional[int],
        typer.Option("--max-papers", "-n", min=1, help="Number of papers to fetch."),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="OpenAI-compatible model for summarization and ranking."),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", help="Path to config YAML file."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", help="OpenAI/Shopify AI Proxy API key (overrides config and env)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show API usage stats."),
    ] = False,
) -> None:
    """Fetch recent papers, summarize them with OpenAI, and rank by relevance.

    Use `lit-agg digest --help` for personalized category digests, and
    `lit-agg profiles` to list configured profiles.
    """
    config = load_config(config_path)

    cat_list = _split_csv(categories) or config.default_categories
    n_papers = max_papers or config.max_papers
    summarize_model = model or config.summarize_model
    rank_model = model or config.rank_model

    # --- Fetch papers ---
    try:
        with console.status("[bold blue]Fetching papers from arxiv..."):
            sources = get_default_sources()
            papers = []
            for source in sources:
                if query:
                    papers.extend(source.search(query, max_results=n_papers))
                else:
                    papers.extend(source.fetch_recent(cat_list, max_results=n_papers))
    except SourceError as e:
        console.print(f"[red]Failed to fetch papers: {e}[/red]")
        raise typer.Exit(1)

    papers = _dedupe_papers(papers)
    if not papers:
        console.print("[red]No papers found.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Fetched {len(papers)} papers.[/green]")

    # --- OpenAI client ---
    client = _client_or_exit(config, api_key)

    # --- Summarize ---
    with console.status("[bold blue]Summarizing papers with OpenAI..."):
        summaries = summarize_papers(
            client,
            papers,
            model=summarize_model,
            batch_size=config.batch_size,
            verbose=verbose,
        )

    console.print(f"[green]Summarized {len(summaries)} papers.[/green]")

    # --- Rank ---
    with console.status("[bold blue]Ranking papers with OpenAI..."):
        ranked = rank_papers(
            client,
            papers,
            summaries,
            model=rank_model,
            query=query,
            verbose=verbose,
        )

    # --- Display ---
    display_results(ranked, query=query)


@command_app.command("profiles")
def list_profiles(
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", help="Path to config YAML file."),
    ] = None,
) -> None:
    """List configured interest profiles."""
    config = load_config(config_path)
    if not config.profiles:
        console.print("[red]No interest profiles configured.[/red]")
        raise typer.Exit(1)

    console.print("[bold]Available interest profiles:[/bold]")
    default_name = config.default_profile or ("physics" if "physics" in config.profiles else None)
    for name in sorted(config.profiles):
        try:
            profile = resolve_profile(config, name)
        except ProfileError as e:
            console.print(f"- [red]{name}[/red]: {e}")
            continue

        categories = ", ".join(profile.default_categories) or "(uses default categories)"
        first_line = shorten(" ".join(profile.description.split()), width=140, placeholder="...")
        default_marker = " [dim](default)[/dim]" if profile.name == default_name else ""
        console.print(f"- [bold]{profile.name}[/bold]{default_marker}")
        console.print(f"  [dim]categories:[/dim] {categories}")
        console.print(
            f"  [dim]weekly:[/dim] top={profile.weekly_top_papers or config.digest_top_papers}, "
            f"candidates={profile.weekly_max_candidates or config.digest_max_candidates}"
        )
        console.print(f"  [dim]focus:[/dim] {first_line}")


@command_app.command("digest")
def digest(
    profile_name: Annotated[
        Optional[str],
        typer.Option("--profile", "-p", help="Interest profile name from config."),
    ] = None,
    categories: Annotated[
        Optional[str],
        typer.Option(
            "--categories",
            "-c",
            help="Comma-separated arxiv categories. Defaults to profile/default config categories.",
        ),
    ] = None,
    since: Annotated[
        str,
        typer.Option("--since", help="Window start: 7d, 1w, 24h, or ISO date."),
    ] = "7d",
    max_candidates: Annotated[
        Optional[int],
        typer.Option("--max-candidates", min=1, help="Candidate papers to fetch before screening."),
    ] = None,
    top: Annotated[
        Optional[int],
        typer.Option("--top", min=1, help="Final number of ranked summaries to display."),
    ] = None,
    summary_pool: Annotated[
        Optional[int],
        typer.Option(
            "--summary-pool",
            min=1,
            help="Number of screened papers to summarize before final ranking. Defaults to 2x --top.",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="OpenAI-compatible model for screening, summarization, and ranking."),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", help="Path to config YAML file."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", help="OpenAI/Shopify AI Proxy API key (overrides config and env)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show API usage stats."),
    ] = False,
) -> None:
    """Create a personalized digest from recent papers in one or more arXiv categories."""
    config = load_config(config_path)
    try:
        profile = resolve_profile(config, profile_name)
    except ProfileError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    cat_list = _split_csv(categories) or profile.default_categories or config.default_categories
    if not cat_list:
        console.print("[red]No arXiv categories configured. Pass --categories or add them to the profile.[/red]")
        raise typer.Exit(1)

    n_candidates = max_candidates or profile.weekly_max_candidates or config.digest_max_candidates
    n_top = top or profile.weekly_top_papers or config.digest_top_papers
    pool_size = summary_pool or max(n_top * 2, n_top)
    start = _parse_since(since)
    end = datetime.now(UTC)

    console.print(
        f"[bold]Digest profile:[/bold] {profile.name}\n"
        f"[bold]Categories:[/bold] {', '.join(cat_list)}\n"
        f"[bold]Requested window:[/bold] {since}\n"
        f"[bold]arXiv submittedDate query:[/bold] {start.date().isoformat()} to {end.date().isoformat()}"
    )

    # --- Fetch candidates ---
    try:
        with console.status("[bold blue]Fetching recent arxiv candidates..."):
            sources = get_default_sources()
            papers = []
            for source in sources:
                papers.extend(
                    source.fetch_recent_window(
                        cat_list,
                        start=start,
                        end=end,
                        max_results=n_candidates,
                    )
                )
    except SourceError as e:
        console.print(f"[red]Failed to fetch papers: {e}[/red]")
        raise typer.Exit(1)

    papers = _dedupe_papers(papers)
    if not papers:
        console.print("[red]No papers found for this digest window.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Fetched {len(papers)} candidate papers.[/green]")

    client = _client_or_exit(config, api_key)
    screen_model = model or config.screen_model
    summarize_model = model or config.summarize_model
    rank_model = model or config.rank_model

    # --- Screen ---
    with console.status("[bold blue]Screening candidates against interest profile..."):
        screenings = screen_papers(
            client,
            papers,
            interests=profile.description,
            model=screen_model,
            batch_size=config.screening_batch_size,
            verbose=verbose,
        )

    paper_map = {p.source_id: p for p in papers}
    shortlisted = [paper_map[s.source_id] for s in screenings if s.source_id in paper_map]
    shortlisted = shortlisted[: min(len(shortlisted), pool_size)]

    if not shortlisted:
        console.print("[red]No papers survived relevance screening.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Shortlisted {len(shortlisted)} papers for summarization.[/green]")

    # --- Summarize the shortlist ---
    with console.status("[bold blue]Summarizing shortlisted papers..."):
        summaries = summarize_papers(
            client,
            shortlisted,
            model=summarize_model,
            batch_size=config.batch_size,
            verbose=verbose,
        )

    # --- Final rank ---
    with console.status("[bold blue]Ranking digest papers..."):
        ranked = rank_papers(
            client,
            shortlisted,
            summaries,
            model=rank_model,
            query=profile.description,
            verbose=verbose,
        )

    display_results(ranked[:n_top], query=f"{profile.name} digest ({since})")


def app() -> None:
    """Dispatch to the legacy search command or newer subcommands."""
    if len(sys.argv) > 1 and sys.argv[1] in _COMMANDS:
        command_app(prog_name="lit-agg")
    else:
        search_app(prog_name="lit-agg")


if __name__ == "__main__":
    app()
