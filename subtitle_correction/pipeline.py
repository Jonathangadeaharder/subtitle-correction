import json
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from rich.console import Console

from .scraper import OpenSubtitlesScraper
from .parser import smart_parse
from .align import compute_alignment_score
from .hallucination_filter import filter_hallucinations

console = Console()

CORRECTION_MIN_ALIGNMENT = 0.5



class PipelineStep:
    PENDING = "pending"
    AUDIO_EXTRACTED = "audio_extracted"
    WHISPER_DONE = "whisper_done"
    SUBTITLE_DOWNLOADED = "subtitle_downloaded"
    ALIGNED = "aligned"
    FAILED = "failed"


class FileMetadata:
    def __init__(self, cache_dir: Path, source_file: str):
        self.cache_dir = cache_dir
        self.source_file = source_file
        self.metadata_path = cache_dir / "metadata.json"
        self._data: dict = {}

    @property
    def status(self) -> str:
        return self._data.get("status", PipelineStep.PENDING)

    @status.setter
    def status(self, value: str) -> None:
        self._data["status"] = value
        self._data["updated_at"] = datetime.now().isoformat()
        self.save()

    @property
    def whisper_lang(self) -> str:
        return self._data.get("whisper_lang", "")

    @whisper_lang.setter
    def whisper_lang(self, value: str) -> None:
        self._data["whisper_lang"] = value

    @property
    def subtitle_lang(self) -> str:
        return self._data.get("subtitle_lang", "")

    @subtitle_lang.setter
    def subtitle_lang(self, value: str) -> None:
        self._data["subtitle_lang"] = value

    @property
    def alignment_score(self) -> float:
        return self._data.get("alignment_score", 0.0)

    @alignment_score.setter
    def alignment_score(self, value: float) -> None:
        self._data["alignment_score"] = value

    @property
    def error(self) -> str:
        return self._data.get("error", "")

    @error.setter
    def error(self, value: str) -> None:
        self._data["error"] = value

    def load(self) -> None:
        if self.metadata_path.exists():
            self._data = json.loads(self.metadata_path.read_text())
        else:
            self._data = {
                "source_file": self.source_file,
                "status": PipelineStep.PENDING,
                "created_at": datetime.now().isoformat(),
            }

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(self._data, indent=2))

    def is_complete(self) -> bool:
        return self.status == PipelineStep.ALIGNED

    def needs_step(self, step: str) -> bool:
        order = [
            PipelineStep.PENDING,
            PipelineStep.AUDIO_EXTRACTED,
            PipelineStep.WHISPER_DONE,
            PipelineStep.SUBTITLE_DOWNLOADED,
            PipelineStep.ALIGNED,
        ]
        current_idx = order.index(self.status) if self.status in order else -1
        step_idx = order.index(step) if step in order else len(order)
        return current_idx < step_idx


def extract_audio(mp4_path: Path, output_wav: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return output_wav


def run_whisper(
    wav_path: Path,
    output_srt: Path,
    output_json: Path | None = None,
    language: str | None = None,
    model: str = "mlx-community/whisper-large-v3-turbo",
    anti_hallucination: bool = True,
) -> Path:
    from mlx_audio.stt.generate import generate_transcription

    kwargs: dict = {}
    if anti_hallucination:
        kwargs.update(
            {
                "condition_on_previous_text": False,
                "hallucination_silence_threshold": 2.0,
                "word_timestamps": True,
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
            }
        )
    if language:
        kwargs["language"] = language

    output_stem = str(output_srt.with_suffix(""))

    segments = generate_transcription(
        model=model,
        audio=str(wav_path),
        output_path=output_stem,
        format="srt",
        verbose=False,
        **kwargs,
    )

    if not output_srt.exists():
        srt_files = list(output_srt.parent.glob(f"{output_srt.stem}*.srt"))
        if srt_files:
            shutil.move(str(srt_files[0]), str(output_srt))
        else:
            raise FileNotFoundError(f"Whisper did not produce SRT: {output_srt}")

    if output_json and hasattr(segments, "segments") and segments.segments:
        result = {"text": getattr(segments, "text", ""), "segments": []}
        for s in segments.segments:
            seg = {
                "text": s.get("text", ""),
                "start": s.get("start", 0),
                "end": s.get("end", 0),
                "no_speech_prob": s.get("no_speech_prob", 0.0),
                "avg_logprob": s.get("avg_logprob", 0.0),
                "compression_ratio": s.get("compression_ratio", 1.0),
            }
            result["segments"].append(seg)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    return output_srt


def run_correction(
    input_srt: Path,
    reference_srt: Path,
    output_srt: Path,
    model_path: str = "runs/subtitle-corrector-4b-fused",
    fused: bool = True,
) -> Path:
    resolved = str(Path(model_path).resolve())
    cmd = [
        "subtitle-correction",
        "correct",
        "--input-file",
        str(input_srt),
        "--reference-file",
        str(reference_srt),
        "--output",
        str(output_srt),
        "--model",
        resolved,
    ]
    if fused:
        cmd.append("--fused")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Subtitle correction failed: {result.stderr}")
    return output_srt


def run_hallucination_filter(
    srt_path: Path,
    audio_path: Path,
    output_srt: Path,
    json_path: Path | None = None,
    lang: str = "en",
    enable_vad: bool = True,
    enable_confidence: bool = True,
    enable_heuristics: bool = True,
) -> Path:
    stats = filter_hallucinations(
        srt_path=srt_path,
        audio_path=audio_path,
        output_path=output_srt,
        json_path=json_path,
        lang=lang,
        enable_vad=enable_vad,
        enable_confidence=enable_confidence,
        enable_heuristics=enable_heuristics,
    )

    dropped = stats.total - stats.kept
    console.print(
        f"  [dim]Hallucination filter: {dropped}/{stats.total} dropped "
        f"(VAD={stats.dropped_vad}, confidence={stats.dropped_confidence}, "
        f"repetition={stats.dropped_repetition}, "
        f"phrase={stats.dropped_hallucination_phrase})[/dim]"
    )

    for reason in stats.reasons[:10]:
        console.print(f"    [dim]{reason}[/dim]")

    return output_srt


def run_opensubtitles_download(
    mp4_name: str, language: str, output_dir: Path, scraper=None
) -> Path | None:
    parsed = smart_parse(mp4_name)
    query = parsed.search_query
    if parsed.episode_label:
        query = f"{query} {parsed.episode_label}"

    own_scraper = scraper is None
    if own_scraper:
        scraper = OpenSubtitlesScraper()

    try:
        results = scraper.search(query, language)
        if not results:
            console.print(f"[yellow]No OpenSubtitles results for: {query}[/yellow]")
            return None

        lang_lower = language.lower()
        lang_filtered = (
            [r for r in results if lang_lower in r.language.lower()]
            if any(r.language for r in results)
            else results
        )
        if not lang_filtered:
            lang_filtered = results

        best = lang_filtered[0]
        for r in lang_filtered:
            if parsed.title.lower() in r.name.lower():
                best = r
                break

        path = scraper.download(best, output_dir)
        if path.exists() and path.stat().st_size > 0:
            return path
    except Exception as e:
        console.print(f"[yellow]OpenSubtitles download failed: {e}[/yellow]")
    finally:
        if own_scraper:
            scraper.close()
    return None


def run_alignment(whisper_srt: Path, subtitle_srt: Path, output_srt: Path) -> Path:
    cmd = [
        "subtitle-correction",
        "align",
        str(whisper_srt),
        str(subtitle_srt),
        "--output",
        str(output_srt),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"subtitle-align failed: {result.stderr}")
    return output_srt


def safe_dirname(filename: str) -> str:
    name = re.sub(r"[^\w.-]", "_", Path(filename).stem)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:100]


def detect_language_from_filename(filename: str) -> str | None:
    lower = filename.lower()
    if any(kw in lower for kw in ["german", "germandub", "deutsch"]):
        return "de"
    if any(kw in lower for kw in ["englisch", "english"]):
        return "en"
    return None


def process_file(
    mp4_path: Path,
    cache_root: Path,
    language: str = "en",
    skip_existing: bool = True,
    scraper=None,
    correct_whisper: bool = True,
    corrector_model: str = "runs/subtitle-corrector-4b-fused",
    filter_hallucinations: bool = True,
) -> dict:
    dir_name = safe_dirname(mp4_path.name)
    work_dir = cache_root / dir_name
    work_dir.mkdir(parents=True, exist_ok=True)

    meta = FileMetadata(work_dir, mp4_path.name)
    meta.load()

    if skip_existing and meta.is_complete():
        return {"status": "skipped", "dir": str(work_dir)}

    wav_path = work_dir / "audio.wav"
    whisper_srt = work_dir / "whisper.srt"
    whisper_json = work_dir / "whisper.json"
    filtered_srt = work_dir / "whisper_filtered.srt"
    corrected_srt = work_dir / "whisper_corrected.srt"
    file_lang = detect_language_from_filename(mp4_path.name)
    whisper_lang = file_lang or language
    detected_lang = meta.whisper_lang or whisper_lang or "en"
    sub_srt = work_dir / f"opensubtitles.{detected_lang}.srt"
    aligned_srt = work_dir / "aligned.srt"

    try:
        if meta.needs_step(PipelineStep.AUDIO_EXTRACTED):
            console.print("  [dim]Extracting audio...[/dim]")
            extract_audio(mp4_path, wav_path)
            meta.status = PipelineStep.AUDIO_EXTRACTED

        if meta.needs_step(PipelineStep.WHISPER_DONE):
            console.print(f"  [dim]Transcribing with Whisper (lang={whisper_lang})...[/dim]")
            run_whisper(wav_path, whisper_srt, output_json=whisper_json, language=whisper_lang)
            meta.whisper_lang = whisper_lang
            sub_srt = work_dir / f"opensubtitles.{whisper_lang}.srt"
            console.print(f"  [dim]Language: {whisper_lang}[/dim]")
            if filter_hallucinations:
                console.print(
                    "  [dim]Filtering hallucinations (VAD + confidence + heuristics)...[/dim]"
                )
                run_hallucination_filter(
                    whisper_srt, wav_path, filtered_srt, json_path=whisper_json, lang=whisper_lang
                )
                shutil.copy(str(filtered_srt), str(whisper_srt))
            meta.status = PipelineStep.WHISPER_DONE

        if meta.needs_step(PipelineStep.SUBTITLE_DOWNLOADED):
            console.print("  [dim]Downloading from OpenSubtitles...[/dim]")
            lang_for_subs = meta.whisper_lang or language or "en"
            downloaded = run_opensubtitles_download(
                mp4_path.name, lang_for_subs, work_dir, scraper=scraper
            )
            if downloaded:
                if downloaded != sub_srt:
                    shutil.move(str(downloaded), str(sub_srt))
                meta.subtitle_lang = lang_for_subs
                meta.status = PipelineStep.SUBTITLE_DOWNLOADED
            else:
                meta.error = "No subtitle found on OpenSubtitles"
                meta.status = PipelineStep.FAILED
                return {"status": "no_subtitle", "dir": str(work_dir)}

        if meta.needs_step(PipelineStep.ALIGNED):
            console.print("  [dim]Aligning timestamps...[/dim]")
            if whisper_srt.exists() and sub_srt.exists():
                run_alignment(whisper_srt, sub_srt, aligned_srt)
                score = compute_alignment_score(whisper_srt, aligned_srt)
                meta.alignment_score = score
                console.print(f"  [dim]Alignment score: {score:.2%}[/dim]")
                if correct_whisper and score >= CORRECTION_MIN_ALIGNMENT:
                    console.print(
                        "  [dim]Correcting Whisper output with MLX model and reference...[/dim]"
                    )
                    run_correction(
                        whisper_srt,
                        aligned_srt,
                        corrected_srt,
                        model_path=corrector_model,
                        fused=True,
                    )
                    shutil.copy(str(corrected_srt), str(whisper_srt))
                    console.print("  [dim]Whisper output corrected[/dim]")
                elif correct_whisper:
                    console.print(
                        f"  [yellow]Skipping correction: alignment score "
                        f"{score:.2%} below threshold "
                        f"{CORRECTION_MIN_ALIGNMENT:.2%}[/yellow]"
                    )
                meta.status = PipelineStep.ALIGNED
            else:
                meta.error = "Missing SRT files for alignment"
                meta.status = PipelineStep.FAILED
                return {"status": "alignment_failed", "dir": str(work_dir)}

    except Exception as e:
        meta.error = str(e)
        meta.status = PipelineStep.FAILED
        console.print(f"  [red]Error: {e}[/red]")
        return {"status": "failed", "error": str(e), "dir": str(work_dir)}

    return {"status": meta.status, "dir": str(work_dir)}
