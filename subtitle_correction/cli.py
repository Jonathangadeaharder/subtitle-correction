import os
import re
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(help="Unified Subtitle Correction Command-Line Interface")

# Sub-typers or sub-commands setup
TV_EPISODE_RE = re.compile(r"[._ -]S\d+E\d+[._ -]", re.IGNORECASE)


def _default_corrector_model() -> Path:
    repo_default = Path(__file__).resolve().parents[1] / "runs" / "subtitle-corrector-4b-fused"
    return Path(os.getenv("SUBTITLE_CORRECTOR_MODEL", str(repo_default)))


# --- ALIGN COMMANDS ---
@app.command(name="align")
def align_cmd(
    whisper_srt: Path = typer.Argument(..., help="Whisper-generated SRT file"),
    subtitle_srt: Path = typer.Argument(..., help="Reference subtitle SRT file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output aligned SRT path"),
    split_penalty: int = typer.Option(10, "--split-penalty", help="alass split penalty (0-1000)"),
) -> None:
    """Align subtitle timestamps to match Whisper SRT."""
    from .align import (
        align_with_alass,
        compute_alignment_score,
        detect_srt_language,
        extract_language_from_filename,
    )

    if not whisper_srt.exists():
        console.print(f"[red]Whisper SRT not found: {whisper_srt}[/red]")
        raise typer.Exit(code=1)
    if not subtitle_srt.exists():
        console.print(f"[red]Subtitle SRT not found: {subtitle_srt}[/red]")
        raise typer.Exit(code=1)

    if output is None:
        output = whisper_srt.parent / f"{whisper_srt.stem}.aligned.srt"

    whisper_lang = detect_srt_language(whisper_srt)
    sub_lang = extract_language_from_filename(subtitle_srt) or detect_srt_language(subtitle_srt)

    console.print(f"Whisper language: [cyan]{whisper_lang}[/cyan]")
    console.print(f"Subtitle language: [cyan]{sub_lang}[/cyan]")

    console.print("\n[bold]Step 1: Aligning with alass...[/bold]")
    try:
        align_with_alass(whisper_srt, subtitle_srt, output, split_penalty)
        console.print(f"[green]alass succeeded: {output}[/green]")
        used_alass = True
    except Exception as e:
        console.print(f"[yellow]alass failed: {e}[/yellow]")
        used_alass = False

    if not used_alass:
        console.print("[red]Alignment failed.[/red]")
        raise typer.Exit(code=1)

    console.print("\n[bold]Step 2: Computing alignment score...[/bold]")
    score = compute_alignment_score(whisper_srt, output)
    console.print(f"Alignment score: {score:.2%}")

    console.print(f"\n[bold green]Done! Aligned SRT: {output}[/bold green]")


# --- SCRAPER COMMANDS ---
@app.command(name="scrape")
def scrape_cmd(
    filename: str = typer.Argument(
        ..., help="Movie or TV episode filename to search subtitles for"
    ),
    language: str = typer.Option("en", "--lang", "-l", help="Language code (en, de, es, etc.)"),
    output_dir: Path = typer.Option(
        Path.home() / "Downloads", "--output-dir", "-o", help="Output directory"
    ),
) -> None:
    """Download matching subtitles from OpenSubtitles for a given file name."""
    from .scraper import OpenSubtitlesScraper
    from .parser import smart_parse

    parsed = smart_parse(filename)
    scraper = OpenSubtitlesScraper()
    try:
        res = scraper.search_and_download(parsed, language=language, output_dir=output_dir)
        if res:
            console.print(f"[bold green]Downloaded subtitle successfully to: {res}[/bold green]")
        else:
            console.print(f"[red]Failed to find or download subtitle for {filename}[/red]")
    finally:
        scraper.close()


# --- PIPELINE COMMANDS ---
@app.command(name="pipeline")
def pipeline_cmd(
    input_dir: Path = typer.Option(
        Path.home() / "Downloads", "--input-dir", "-i", help="Directory with MP4 files"
    ),
    lang: str = typer.Option(
        "en", "--lang", "-l", help="Language code (fallback if auto-detect fails)"
    ),
    file: Path | None = typer.Option(None, "--file", "-f", help="Process single file"),
    no_skip: bool = typer.Option(False, "--no-skip", help="Re-process already completed files"),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Cache directory"),
    tv_only: bool = typer.Option(
        True, "--tv-only/--all", help="Only process TV episodes (SxxExx pattern)"
    ),
    correct: bool = typer.Option(True, "--correct/--no-correct", help="Apply MLX corrector to Whisper output"),
    corrector_model: Path = typer.Option(
        _default_corrector_model(), "--corrector-model", help="Path to fused MLX corrector model"
    ),
    filter_hallucinations: bool = typer.Option(
        True,
        "--filter-hallucinations/--no-filter-hallucinations",
        help="Filter Whisper hallucinations (VAD + confidence + heuristics)",
    ),
) -> None:
    """Run the batch processing pipeline (Audio extraction + Whisper + OpenSubtitles + Alignment)."""
    from .pipeline import process_file, detect_language_from_filename
    from .scraper import OpenSubtitlesScraper

    cache_root = cache_dir or input_dir / ".subcache"
    cache_root.mkdir(parents=True, exist_ok=True)

    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(code=1)
        mp4_files = [file]
    else:
        mp4_files = sorted(input_dir.glob("*.mp4"))
        if tv_only:
            mp4_files = [f for f in mp4_files if TV_EPISODE_RE.search(f.name)]
        if lang == "en":
            before = len(mp4_files)
            mp4_files = [f for f in mp4_files if detect_language_from_filename(f.name) != "de"]
            skipped = before - len(mp4_files)
            if skipped:
                console.print(f"[dim]Skipping {skipped} non-English file(s)[/dim]")

        seen = set()
        deduped = []
        for f in mp4_files:
            base = f.name.replace("_compressed", "")
            if base not in seen:
                seen.add(base)
                deduped.append(f)
        mp4_files = deduped

    if not mp4_files:
        console.print(f"[yellow]No MP4 files found in {input_dir}[/yellow]")
        return

    console.print(f"[bold]Found {len(mp4_files)} video files[/bold]")

    scraper = OpenSubtitlesScraper()
    stats = {"completed": 0, "failed": 0, "skipped": 0}

    try:
        for i, mp4 in enumerate(mp4_files, 1):
            console.print(f"\n[bold cyan][{i}/{len(mp4_files)}] {mp4.name}[/bold cyan]")
            result = process_file(
                mp4,
                cache_root,
                language=lang,
                skip_existing=not no_skip,
                scraper=scraper,
                correct_whisper=correct,
                corrector_model=str(corrector_model),
                filter_hallucinations=filter_hallucinations,
            )
            status = result["status"]
            if status == "skipped":
                stats["skipped"] += 1
            elif status == "aligned":
                stats["completed"] += 1
                console.print(f"  [green]Done ({status})[/green]")
            else:
                stats["failed"] += 1
                console.print(f"  [red]Failed: {result.get('error', status)}[/red]")
    finally:
        scraper.close()

    console.print(
        f"\n[bold]Pipeline complete: {stats['completed']} done, {stats['failed']} failed, {stats['skipped']} skipped[/bold]"
    )


@app.command(name="pipeline-status")
def pipeline_status_cmd(
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Cache directory"),
    input_dir: Path = typer.Option(
        Path.home() / "Downloads", "--input-dir", "-i", help="Input directory"
    ),
) -> None:
    """Show batch pipeline status of cached runs."""
    from .pipeline import FileMetadata, PipelineStep

    cache_root = cache_dir or input_dir / ".subcache"
    if not cache_root.exists():
        console.print("[yellow]No cache directory found[/yellow]")
        return

    subdirs = sorted(d for d in cache_root.iterdir() if d.is_dir())
    if not subdirs:
        console.print("[yellow]No processed files[/yellow]")
        return

    table = Table(title="Pipeline Status")
    table.add_column("File", style="cyan", max_width=50)
    table.add_column("Status", style="bold")
    table.add_column("Whisper Lang", style="green")
    table.add_column("Sub Lang", style="green")
    table.add_column("Align Score", style="yellow")
    table.add_column("Error", style="red", max_width=30)

    status_colors = {
        PipelineStep.PENDING: "dim",
        PipelineStep.AUDIO_EXTRACTED: "yellow",
        PipelineStep.WHISPER_DONE: "yellow",
        PipelineStep.SUBTITLE_DOWNLOADED: "yellow",
        PipelineStep.ALIGNED: "green",
        PipelineStep.FAILED: "bold red",
    }

    for d in subdirs:
        meta = FileMetadata(d, d.name)
        meta.load()
        color = status_colors.get(meta.status, "white")
        table.add_row(
            d.name,
            f"[{color}]{meta.status}[/{color}]",
            meta.whisper_lang or "-",
            meta.subtitle_lang or "-",
            f"{meta.alignment_score:.0%}" if meta.alignment_score else "-",
            meta.error[:30] if meta.error else "",
        )

    console.print(table)


# --- CORRECTION COMMANDS ---
@app.command(name="correct")
def correct_cmd(
    text: str = typer.Option(None, "--text", "-t", help="Text to correct"),
    input_file: Path = typer.Option(None, "--input-file", "-i", help="SRT file to correct"),
    reference_file: Path = typer.Option(
        None, "--reference-file", "-r", help="Reference aligned SRT file"
    ),
    model: str = typer.Option(
        "mlx-community/gemma-4-e4b-it-4bit", "--model", "-m", help="Base model name or path"
    ),
    adapter_path: Path = typer.Option(None, "--adapter-path", "-a", help="LoRA adapter path"),
    fused: bool = typer.Option(
        True, "--fused/--adapter", help="Use fused model or separate adapter"
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output corrected file"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens to generate"),
    temp: float = typer.Option(0.1, "--temp", help="Temperature"),
    lang: str = typer.Option("en", "--lang", help="Filter for only this language"),
) -> None:
    """Run inference to correct speech recognition errors in text or SRT files."""
    from .inference import correct_file_impl, SubtitleCorrector

    if text:
        corrector = SubtitleCorrector(model, adapter_path, fused=fused, temp=temp)
        res = corrector.correct_line(text, max_tokens=max_tokens)
        console.print(f"[bold green]Corrected:[/bold green] {res}")
    elif input_file:
        correct_file_impl(
            input_file, output, model, adapter_path, fused, max_tokens, temp, lang, reference_file
        )
    else:
        console.print("[red]Provide --text or --input-file[/red]")
        raise typer.Exit(code=1)


@app.command(name="srt-from-reference")
def srt_from_reference_cmd(
    reference: Path = typer.Option(
        ..., "--reference", "-r", help="Reference prose or markdown file"
    ),
    whisper_srt: Path = typer.Option(
        ..., "--whisper-srt", "-w", help="Whisper SRT with cue timing"
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output corrected SRT path"),
) -> None:
    """Deterministic reference alignment onto Whisper cue timing (no LLM)."""
    from .srt_from_reference import align_reference_to_srt

    align_reference_to_srt(reference.read_text(encoding="utf-8"), whisper_srt, out)
    console.print(f"[green]Wrote {out}[/green]")


@app.command(name="correct-reference-free")
def correct_reference_free_cmd(
    input_srt: Path = typer.Option(..., "--input", "-i", help="Whisper SRT to correct"),
    output: Path = typer.Option(..., "--output", "-o", help="Output SRT path"),
    model: str = typer.Option(
        "mlx-community/gemma-4-e4b-it-4bit", "--model", "-m", help="Base instruct model"
    ),
    temp: float = typer.Option(0.0, "--temp", help="Sampling temperature"),
) -> None:
    """Correct German Whisper subtitles using neighboring cue context (no reference)."""
    from .reference_free import correct_srt

    changed, total = correct_srt(input_srt, output, model_name=model, temp=temp)
    console.print(f"[green]Done: {changed}/{total} cues changed → {output}[/green]")


if __name__ == "__main__":
    app()
