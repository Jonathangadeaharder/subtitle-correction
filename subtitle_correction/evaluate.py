import json
import re
import subprocess
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def srt_time_to_seconds(srt_time_str: str) -> float:
    # Format: HH:MM:SS,mmm or HH:MM:SS.mmm
    srt_time_str = srt_time_str.replace(".", ",")
    h, m, s_ms = srt_time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def Levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return Levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
        prev = curr
    return prev[-1]


def calculate_wer(reference: str, hypothesis: str) -> float:
    # Remove all punctuation and lowercase
    r_clean = re.sub(r'[^\w\s]', '', reference.lower()).strip()
    h_clean = re.sub(r'[^\w\s]', '', hypothesis.lower()).strip()
    
    r = r_clean.split()
    h = h_clean.split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / len(r)



def create_dataset_impl(
    subcache_dir: Path = Path.home() / "Downloads" / ".subcache",
    output_dir: Path = Path(__file__).parent.parent / "evaluation",
    max_slices: int = 200,
) -> None:
    """Create evaluation dataset by extracting audio slices from cache."""
    import pysrt
    from .align import generate_training_pairs, compute_alignment_score

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_slices_dir = output_dir / "audio_slices"
    audio_slices_dir.mkdir(parents=True, exist_ok=True)

    if not subcache_dir.exists():
        console.print(f"[red]Subcache directory not found: {subcache_dir}[/red]")
        raise typer.Exit(code=1)

    subdirs = sorted(d for d in subcache_dir.iterdir() if d.is_dir())
    if not subdirs:
        console.print("[yellow]No processed folders in subcache[/yellow]")
        return

    console.print(f"[bold]Scanning {len(subdirs)} subcache folders for evaluation slices...[/bold]")

    all_pairs = []

    for d in subdirs:
        audio_wav = d / "audio.wav"
        aligned_srt = d / "aligned.srt"
        whisper_srt = d / "whisper.srt"
        opensubtitles_srt = next(d.glob("opensubtitles.*.srt"), None)

        if not (audio_wav.exists() and aligned_srt.exists() and whisper_srt.exists() and opensubtitles_srt):
            continue

        score = compute_alignment_score(whisper_srt, aligned_srt)
        if score < 0.5:
            continue

        # Get aligned pairs
        pairs = generate_training_pairs(whisper_srt, aligned_srt, source_file=d.name, alignment_score=score)
        
        # We also want to harvest some identical pairs to test model's stability
        # Let's extract identity pairs manually
        w_subs = pysrt.open(str(whisper_srt))
        a_subs = pysrt.open(str(aligned_srt))
        
        # Match exactly identical text segments
        import bisect
        a_starts = [a.start.ordinal for a in a_subs]
        for w in w_subs:
            w_start = w.start.ordinal
            idx = bisect.bisect_left(a_starts, w_start)
            for j in (idx - 1, idx):
                if 0 <= j < len(a_subs):
                    a = a_subs[j]
                    if abs(a.start.ordinal - w_start) <= 2000:
                        w_text = w.text.strip()
                        a_text = a.text.strip()
                        if w_text and a_text and w_text == a_text and len(w_text) > 10:
                            pairs.append({
                                "whisper_text": w_text,
                                "corrected_text": a_text,
                                "source_file": d.name,
                                "alignment_score": score,
                                "timestamp_start": str(a.start),
                                "timestamp_end": str(a.end),
                            })
                            break
        
        all_pairs.extend(pairs)

    if not all_pairs:
        console.print("[red]No pairs found for evaluation[/red]")
        return

    # Select representative pairs: half with errors, half identical
    error_pairs = [p for p in all_pairs if p["whisper_text"] != p["corrected_text"]]
    identity_pairs = [p for p in all_pairs if p["whisper_text"] == p["corrected_text"]]

    console.print(f"Found {len(error_pairs)} candidates with speech transcription errors")
    console.print(f"Found {len(identity_pairs)} candidates with correct transcriptions (identities)")

    half_max = max_slices // 2
    selected_errors = error_pairs[:half_max]
    selected_identities = identity_pairs[:half_max]
    selected_pairs = selected_errors + selected_identities

    console.print(f"[bold]Slicing {len(selected_pairs)} audio segments...[/bold]")

    dataset_entries = []

    for i, pair in enumerate(selected_pairs, 1):
        source_dir = subcache_dir / pair["source_file"]
        audio_wav = source_dir / "audio.wav"
        
        slice_id = f"slice_{i:04d}"
        slice_file = audio_slices_dir / f"{slice_id}.wav"
        
        start_sec = srt_time_to_seconds(pair["timestamp_start"])
        end_sec = srt_time_to_seconds(pair["timestamp_end"])
        duration = end_sec - start_sec
        
        if duration <= 0:
            continue
            
        # ffmpeg audio cropping
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(audio_wav),
            "-c", "copy",
            str(slice_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            dataset_entries.append({
                "id": slice_id,
                "audio_path": f"audio_slices/{slice_file.name}",
                "whisper_text": pair["whisper_text"],
                "ground_truth": pair["corrected_text"],
                "source_file": pair["source_file"],
                "start_time": pair["timestamp_start"],
                "end_time": pair["timestamp_end"],
                "has_error": pair["whisper_text"] != pair["corrected_text"]
            })
        if i % 20 == 0 or i == len(selected_pairs):
            console.print(f"  Sliced {i}/{len(selected_pairs)}")

    # Write dataset.jsonl
    dataset_file = output_dir / "dataset.jsonl"
    with open(dataset_file, "w") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")

    console.print(f"[bold green]Dataset created successfully at {dataset_file} with {len(dataset_entries)} slices![/bold green]")


def evaluate_model_impl(
    model: str = "mlx-community/Qwen3-4B-4bit",
    adapter_path: Path | None = None,
    fused: bool = False,
    dataset_path: Path = Path(__file__).parent.parent / "evaluation" / "dataset.jsonl",
) -> None:
    """Evaluate subtitle corrector model on the generated audio slice dataset."""
    if not dataset_path.exists():
        console.print(f"[red]Dataset file not found: {dataset_path}[/red]")
        raise typer.Exit(code=1)

    from .inference import SubtitleCorrector

    entries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        console.print("[red]Evaluation dataset is empty[/red]")
        return

    console.print(f"[bold]Evaluating model on {len(entries)} test cases...[/bold]")
    
    # Initialize corrector
    corrector = SubtitleCorrector(model, adapter_path, fused=fused)

    stats = {
        "error_cases": {
            "total": 0,
            "corrected": 0,
            "unchanged": 0,
            "worsened": 0,
        },
        "identity_cases": {
            "total": 0,
            "unchanged": 0,
            "changed": 0,
        },
        "wer_before": [],
        "wer_after": [],
    }

    table = Table(title="Sample Corrections")
    table.add_column("ID", style="dim")
    table.add_column("Whisper (Input)", style="red")
    table.add_column("Reference", style="yellow")
    table.add_column("Model Output", style="green")
    table.add_column("Ground Truth", style="cyan")

    sample_count = 0

    for i, entry in enumerate(entries, 1):
        whisper = entry["whisper_text"]
        target = entry["ground_truth"]
        # Reference = aligned OpenSubtitles text (= ground_truth in eval dataset)
        reference = entry.get("reference", target)

        corrected = corrector.correct_line(whisper, reference=reference)

        wer_b = calculate_wer(target, whisper)
        wer_a = calculate_wer(target, corrected)

        
        stats["wer_before"].append(wer_b)
        stats["wer_after"].append(wer_a)

        def clean_str(s):
            return re.sub(r'[^\w\s]', '', s.lower()).strip()
        
        w_clean = clean_str(whisper)
        t_clean = clean_str(target)
        c_clean = clean_str(corrected)

        if entry["has_error"]:
            stats["error_cases"]["total"] += 1
            if c_clean == t_clean:
                stats["error_cases"]["corrected"] += 1
            elif c_clean == w_clean:
                stats["error_cases"]["unchanged"] += 1
            else:
                dist_before = Levenshtein_distance(w_clean, t_clean)
                dist_after = Levenshtein_distance(c_clean, t_clean)
                if dist_after < dist_before:
                    stats["error_cases"]["corrected"] += 1
                else:
                    stats["error_cases"]["worsened"] += 1
            if sample_count < 10:
                    table.add_row(entry["id"], whisper, reference, corrected, target)
                    sample_count += 1

        else:
            stats["identity_cases"]["total"] += 1
            if c_clean == w_clean:
                stats["identity_cases"]["unchanged"] += 1
            else:
                stats["identity_cases"]["changed"] += 1


        if i % 10 == 0 or i == len(entries):
            console.print(f"  Processed {i}/{len(entries)}...")

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Summarize results
    total_wer_b = sum(stats["wer_before"]) / len(stats["wer_before"])
    total_wer_a = sum(stats["wer_after"]) / len(stats["wer_after"])
    
    console.print("[bold cyan]=========================================[/bold cyan]")
    console.print("[bold cyan]          EVALUATION REPORT              [/bold cyan]")
    console.print("[bold cyan]=========================================[/bold cyan]")
    
    console.print(f"Total Test Cases: {len(entries)}")
    console.print(f"Word Error Rate (WER) Before: {total_wer_b:.2%}")
    console.print(f"Word Error Rate (WER) After:  {total_wer_a:.2%}")
    console.print(f"Relative WER Improvement:     {(total_wer_b - total_wer_a) / (total_wer_b + 1e-6):.2%}")
    console.print("")
    
    err_stats = stats["error_cases"]
    console.print("[bold]Error Correction (Targeting Speech Errors):[/bold]")
    console.print(f"  Total Error Cases: {err_stats['total']}")
    console.print(f"  Successfully Corrected/Improved: {err_stats['corrected']} ({err_stats['corrected']/(err_stats['total']+1e-6):.2%})")
    console.print(f"  Unchanged: {err_stats['unchanged']} ({err_stats['unchanged']/(err_stats['total']+1e-6):.2%})")
    console.print(f"  Worsened: {err_stats['worsened']} ({err_stats['worsened']/(err_stats['total']+1e-6):.2%})")
    console.print("")
    
    id_stats = stats["identity_cases"]
    console.print("[bold]Stability (Identity Preservation):[/bold]")
    console.print(f"  Total Identity Cases: {id_stats['total']}")
    console.print(f"  Preserved (Unchanged): {id_stats['unchanged']} ({id_stats['unchanged']/(id_stats['total']+1e-6):.2%})")
    console.print(f"  Corrupted (Wrongly Changed): {id_stats['changed']} ({id_stats['changed']/(id_stats['total']+1e-6):.2%})")
    console.print("[bold cyan]=========================================[/bold cyan]")
