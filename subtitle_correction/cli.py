import re
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(help="Unified Subtitle Correction Command-Line Interface")

# Sub-typers or sub-commands setup
TV_EPISODE_RE = re.compile(r'[._ -]S\d+E\d+[._ -]', re.IGNORECASE)

# --- ALIGN COMMANDS ---
@app.command(name="align")
def align_cmd(
    whisper_srt: Path = typer.Argument(..., help="Whisper-generated SRT file"),
    subtitle_srt: Path = typer.Argument(..., help="Reference subtitle SRT file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output aligned SRT path"),
    split_penalty: int = typer.Option(10, "--split-penalty", help="alass split penalty (0-1000)"),
    skip_pairs: bool = typer.Option(False, "--skip-pairs", help="Skip training pair generation"),
    pairs_output: Path = typer.Option(Path("training_pairs.jsonl"), "--pairs-output", help="Training pairs JSONL path"),
    min_score: float = typer.Option(0.5, "--min-score", help="Minimum alignment score for pairs"),
) -> None:
    """Align subtitle timestamps to match Whisper SRT."""
    from .align import align_with_alass, compute_alignment_score, generate_training_pairs, append_pairs_to_jsonl, detect_srt_language, extract_language_from_filename
    
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

    if not skip_pairs and whisper_lang == sub_lang:
        console.print("\n[bold]Step 3: Generating training pairs...[/bold]")
        pairs = generate_training_pairs(whisper_srt, output, source_file=whisper_srt.stem, min_score=min_score, alignment_score=score)
        count = append_pairs_to_jsonl(pairs, pairs_output)
        console.print(f"[green]Added {count} pairs to {pairs_output}[/green]")
    elif whisper_lang != sub_lang:
        console.print(f"\n[yellow]Skipping pairs (language mismatch: {whisper_lang} vs {sub_lang})[/yellow]")

    console.print(f"\n[bold green]Done! Aligned SRT: {output}[/bold green]")


# --- SCRAPER COMMANDS ---
@app.command(name="scrape")
def scrape_cmd(
    filename: str = typer.Argument(..., help="Movie or TV episode filename to search subtitles for"),
    language: str = typer.Option("en", "--lang", "-l", help="Language code (en, de, es, etc.)"),
    output_dir: Path = typer.Option(Path.home() / "Downloads", "--output-dir", "-o", help="Output directory"),
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
    input_dir: Path = typer.Option(Path.home() / "Downloads", "--input-dir", "-i", help="Directory with MP4 files"),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code (fallback if auto-detect fails)"),
    file: Path | None = typer.Option(None, "--file", "-f", help="Process single file"),
    no_skip: bool = typer.Option(False, "--no-skip", help="Re-process already completed files"),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Cache directory"),
    tv_only: bool = typer.Option(True, "--tv-only/--all", help="Only process TV episodes (SxxExx pattern)"),
    correct: bool = typer.Option(False, "--correct", help="Apply MLX corrector to Whisper output"),
    corrector_model: str = typer.Option("/Users/jonathangadeaharder/projects/vidiomtm/subtitle-correction/runs/subtitle-corrector-4b-fused", "--corrector-model", help="Path to fused MLX corrector model"),
    filter_hallucinations: bool = typer.Option(True, "--filter-hallucinations/--no-filter-hallucinations", help="Filter Whisper hallucinations (VAD + confidence + heuristics)"),
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
                mp4, cache_root, language=lang,
                skip_existing=not no_skip, scraper=scraper,
                correct_whisper=correct, corrector_model=corrector_model,
                filter_hallucinations=filter_hallucinations,
            )
            status = result["status"]
            if status == "skipped":
                stats["skipped"] += 1
            elif status in ("paired", "aligned"):
                stats["completed"] += 1
                console.print(f"  [green]Done ({status})[/green]")
            else:
                stats["failed"] += 1
                console.print(f"  [red]Failed: {result.get('error', status)}[/red]")
    finally:
        scraper.close()

    console.print(f"\n[bold]Pipeline complete: {stats['completed']} done, {stats['failed']} failed, {stats['skipped']} skipped[/bold]")


@app.command(name="pipeline-status")
def pipeline_status_cmd(
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Cache directory"),
    input_dir: Path = typer.Option(Path.home() / "Downloads", "--input-dir", "-i", help="Input directory"),
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
        PipelineStep.PAIRED: "bold green",
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


# --- FINETUNING / CORRECTION COMMANDS ---
@app.command(name="prepare")
def prepare_cmd(
    input_file: Path = typer.Option(..., "--input", "-i", help="Training pairs JSONL file"),
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-o", help="Output directory"),
    val_split: float = typer.Option(0.1, "--val-split", help="Validation split ratio"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    augment: int = typer.Option(3, "--augment", help="Number of synthetic corruptions per correct text"),
    identity_ratio: float = typer.Option(0.15, "--identity-ratio", help="Ratio of identity examples"),
) -> None:
    """Prepare and augment training dataset for instruction tuning."""
    from .prepare_data import prepare_data_impl
    prepare_data_impl(input_file, output_dir, val_split, seed, augment, identity_ratio)


@app.command(name="train")
def train_cmd() -> None:
    """Train the correction model using MLX LoRA fine-tuning."""
    from .train import train_impl
    train_impl()


@app.command(name="correct")
def correct_cmd(
    text: str = typer.Option(None, "--text", "-t", help="Text to correct"),
    input_file: Path = typer.Option(None, "--input-file", "-i", help="SRT file to correct"),
    reference_file: Path = typer.Option(None, "--reference-file", "-r", help="Reference aligned SRT file"),
    model: str = typer.Option("mlx-community/gemma-4-e4b-it-4bit", "--model", "-m", help="Base model name or path"),
    adapter_path: Path = typer.Option(None, "--adapter-path", "-a", help="LoRA adapter path"),
    fused: bool = typer.Option(True, "--fused/--adapter", help="Use fused model or separate adapter"),
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
        correct_file_impl(input_file, output, model, adapter_path, fused, max_tokens, temp, lang, reference_file)
    else:
        console.print("[red]Provide --text or --input-file[/red]")
        raise typer.Exit(code=1)


# --- EVALUATION COMMANDS ---
@app.command(name="create-dataset")
def create_dataset_cmd(
    subcache_dir: Path = typer.Option(Path.home() / "Downloads" / ".subcache", "--subcache-dir", help="Subcache folder"),
    output_dir: Path = typer.Option(Path(__file__).parent.parent / "evaluation", "--output-dir", help="Output directory"),
    max_slices: int = typer.Option(200, "--max-slices", help="Maximum slices to crop"),
) -> None:
    """Create evaluation dataset from cached runs by extracting audio slices."""
    from .evaluate import create_dataset_impl
    create_dataset_impl(subcache_dir, output_dir, max_slices)


@app.command(name="evaluate")
def evaluate_cmd(
    model: str = typer.Option("mlx-community/gemma-4-e4b-it-4bit", "--model", "-m", help="Base model"),




    adapter_path: Path = typer.Option(None, "--adapter-path", "-a", help="LoRA adapter path"),
    fused: bool = typer.Option(True, "--fused/--adapter", help="Use fused model or adapter"),
    dataset_path: Path = typer.Option(Path(__file__).parent.parent / "evaluation" / "dataset.jsonl", "--dataset-path", help="Dataset path"),
) -> None:
    """Evaluate correction accuracy on the created slice dataset."""
    from .evaluate import evaluate_model_impl
    evaluate_model_impl(model, adapter_path, fused, dataset_path)


if __name__ == "__main__":
    app()
