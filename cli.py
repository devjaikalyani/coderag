#!/usr/bin/env python3
"""
cli.py  — Command-line interface for CodeRAG
--------------------------------------------
Usage:
  python scripts/cli.py ingest --path ./my_project
  python scripts/cli.py ingest --github https://github.com/owner/repo
  python scripts/cli.py query "How does authentication work?"
  python scripts/cli.py chat          (interactive REPL)
  python scripts/cli.py eval --dataset data/eval.json
"""

import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import CodeRAGPipeline

console = Console()


def get_pipeline() -> CodeRAGPipeline:
    with console.status("[bold green]Initializing pipeline…"):
        return CodeRAGPipeline.from_config()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """🔍 CodeRAG — RAG-powered code assistant"""
    pass


@cli.command()
@click.option("--path", default=None, help="Local directory or file path")
@click.option("--github", default=None, help="GitHub repo URL")
@click.option("--branch", default="main", help="GitHub branch")
@click.option("--text", default=None, help="Raw text to ingest")
def ingest(path, github, branch, text):
    """Ingest documents into the vector index."""
    p = get_pipeline()
    if path:
        console.print(f"[cyan]Ingesting directory:[/cyan] {path}")
        p.ingest_directory(path)
    elif github:
        console.print(f"[cyan]Ingesting GitHub repo:[/cyan] {github} ({branch})")
        p.ingest_github(github, branch=branch)
    elif text:
        p.ingest_text(text)
    else:
        console.print("[red]Provide --path, --github, or --text[/red]")
        sys.exit(1)

    total = p.faiss_index.index.ntotal
    console.print(f"[bold green]✅ Done! Index now contains {total} chunks.[/bold green]")


@cli.command()
@click.argument("question")
@click.option("--no-faith", is_flag=True, default=False, help="Skip faithfulness check")
def query(question, no_faith):
    """Ask a single question."""
    p = get_pipeline()
    if p.faiss_index is None:
        console.print("[red]No index found. Run `ingest` first.[/red]")
        sys.exit(1)

    with console.status("[bold green]Thinking…"):
        response = p.query(question, check_faithfulness=not no_faith)

    console.print(Panel(Markdown(response.answer), title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))

    # Sources table
    if response.sources:
        table = Table(title="Sources", show_header=True)
        table.add_column("Source", style="dim")
        table.add_column("Lines", style="dim", width=10)
        table.add_column("Lang", width=10)
        table.add_column("Score", width=8)
        for r in response.sources:
            table.add_row(
                r.chunk.source,
                f"{r.chunk.start_line}-{r.chunk.end_line}",
                r.chunk.language or "?",
                f"{r.rerank_score:.3f}",
            )
        console.print(table)

    if response.faithfulness:
        color = "green" if response.faithfulness.is_faithful else "red"
        console.print(f"[{color}]Faithfulness: {response.faithfulness.score:.2%}[/{color}]")


@cli.command()
def chat():
    """Interactive chat REPL with conversation memory."""
    p = get_pipeline()
    if p.faiss_index is None:
        console.print("[red]No index found. Run `ingest` first.[/red]")
        sys.exit(1)

    console.print(Panel(
        "[bold]CodeRAG Interactive Chat[/bold]\nType [cyan]exit[/cyan] or [cyan]quit[/cyan] to stop. "
        "[cyan]clear[/cyan] to reset history.",
        border_style="green",
    ))

    while True:
        try:
            question = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "clear":
            p.clear_history()
            console.print("[dim]History cleared.[/dim]")
            continue

        with console.status("[bold green]Thinking…"):
            response = p.query(question)

        console.print(f"\n[bold cyan]CodeRAG:[/bold cyan]")
        console.print(Markdown(response.answer))

        if response.faithfulness:
            color = "green" if response.faithfulness.is_faithful else "yellow"
            console.print(f"[{color}][Faith: {response.faithfulness.score:.0%}][/{color}]\n")


@cli.command()
@click.option("--dataset", required=True, help="Path to JSON eval dataset")
def eval(dataset):
    """Run evaluation on a QA dataset."""
    from src.evaluation.evaluator import RAGEvaluator, load_eval_dataset
    p = get_pipeline()
    samples = load_eval_dataset(dataset)
    evaluator = RAGEvaluator(p)
    df = evaluator.evaluate(samples)
    evaluator.save_results(df)
    console.print(df.to_string())


if __name__ == "__main__":
    cli()
