# subtitle-correction

Runtime correction of **Whisper subtitle transcriptions** on Apple Silicon (MLX). The training,
data-generation, and active-learning scaffolding has been removed — this repo is
**runtime-correction only**.

The corrector is a fused MLX LoRA adapter on `mlx-community/gemma-4-e4b-it-4bit` (rank 64). It
keeps Whisper's *timing* and fixes *content* (mishearings, hallucinations, casing, punctuation)
from a reference, preserving formatting (line breaks, `♪` music marks).

## Three modes

### 1. OpenSubtitles mode (full pipeline)

Audio → Whisper → hallucination filter → OpenSubtitles fetch → alass align → alignment-score
gate → LLM correct. Correction runs by default when the alignment score clears
`CORRECTION_MIN_ALIGNMENT` (0.5); below it the reference is treated as untrusted and the LLM is
skipped.

```bash
subtitle-correction pipeline --input-dir ~/Downloads --lang en --file movie.mp4
subtitle-correction pipeline-status --input-dir ~/Downloads
subtitle-correction scrape "Movie.2020.mp4" --lang en
subtitle-correction align whisper.srt reference.srt --output aligned.srt
```

### 2. BYO-reference mode

You supply a reference (an aligned SRT, or prose/markdown). Two paths:

- **LLM correct** — the fine-tuned model rewrites each cue against the reference:
  ```bash
  subtitle-correction correct --input-file whisper.srt --reference-file reference.srt \
      --output corrected.srt --model runs/subtitle-corrector-4b-fused
  ```
- **Deterministic merge** — no LLM; `difflib.SequenceMatcher` slices the reference onto Whisper
  cue timing. Every cue keeps its index/timestamp; only the text is replaced, verbatim:
  ```bash
  subtitle-correction srt-from-reference --reference narration.md --whisper-srt whisper.srt \
      --out corrected.srt
  ```

### 3. Reference-free mode

No reference at all — a context LLM corrects **German** cues using a ±2-cue window (no model
weights beyond the base instruct model):

```bash
subtitle-correction correct-reference-free --input whisper.srt --output corrected.srt
```

## Layout

```text
subtitle_correction/
    cli.py                 # pipeline, pipeline-status, scrape, align, correct,
                           # srt-from-reference, correct-reference-free
    pipeline.py            # OpenSubtitles-mode orchestrator + alignment-score gate
    align.py               # alass/ffsubsync alignment, compute_alignment_score, SRT lang detect
    inference.py           # MLX LLM corrector (reference mode)
    reference_free.py      # context-window corrector (German, reference-free)
    srt_from_reference.py  # deterministic reference merge (no LLM)
    hallucination_filter.py, parser.py, scraper.py, formatter.py, cache.py, models.py
```

## Notes

- Model checkpoints are **not committed** — fused weights are large. Keep the fused model at
  `runs/subtitle-corrector-4b-fused`, or point `SUBTITLE_CORRECTOR_MODEL` at it.
- All commands run via `uv run subtitle-correction <command>` (see `pyproject.toml`).
