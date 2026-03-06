"""CLI entry point for lit-agg."""

from typing import Annotated, Optional

import typer
from rich.console import Console

from lit_agg.claude.client import get_client
from lit_agg.claude.ranker import rank_papers
from lit_agg.claude.summarizer import summarize_papers
from lit_agg.config import load_config
from lit_agg.display import display_results
from lit_agg.sources import get_default_sources

app = typer.Typer(
    name="lit-agg",
    help="Fetch, summarize, and rank recent research papers using Claude.",
    no_args_is_help=False,
)
console = Console()


@app.command()
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
        typer.Option("--max-papers", "-n", help="Number of papers to fetch."),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Claude model for summarization and ranking."),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", help="Path to config YAML file."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", help="Anthropic API key (overrides config and env)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show thinking output and API usage stats."),
    ] = False,
) -> None:
    """Fetch recent papers, summarize them with Claude, and rank by relevance."""
    config = load_config(config_path)

    cat_list = (
        [c.strip() for c in categories.split(",")]
        if categories
        else config.default_categories
    )
    n_papers = max_papers or config.max_papers
    summarize_model = model or config.summarize_model
    rank_model = model or config.rank_model

    # --- Fetch papers ---
    with console.status("[bold blue]Fetching papers from arxiv..."):
        sources = get_default_sources()
        papers = []
        for source in sources:
            if query:
                papers.extend(source.search(query, max_results=n_papers))
            else:
                papers.extend(source.fetch_recent(cat_list, max_results=n_papers))

    if not papers:
        console.print("[red]No papers found.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Fetched {len(papers)} papers.[/green]")

    # --- Claude client ---
    try:
        client = get_client(config, api_key=api_key)
    except Exception as e:
        console.print(f"[red]Failed to create API client: {e}[/red]")
        console.print(
            "[dim]Set ANTHROPIC_API_KEY, use --api-key, or configure api_key_command in config.[/dim]"
        )
        raise typer.Exit(1)

    # --- Summarize ---
    with console.status("[bold blue]Summarizing papers with Claude..."):
        summaries = summarize_papers(
            client,
            papers,
            model=summarize_model,
            batch_size=config.batch_size,
            verbose=verbose,
        )

    console.print(f"[green]Summarized {len(summaries)} papers.[/green]")

    # --- Rank ---
    with console.status("[bold blue]Ranking papers with Claude..."):
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
