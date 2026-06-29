# subtitle-correction

Fine-tuning a small LLM to **correct Whisper subtitle transcriptions against a reference**, and
the data pipeline + active-learning loop that produced the training data.

## The task

The model takes two inputs and produces one:

- **Whisper transcription** — correct *timing*, but contains mishearings and hallucinations.
- **Reference subtitle** — correct *content*, but may be phrased differently or slightly out of sync.
- **Output** — the corrected subtitle: Whisper's timing, with mishearings and hallucinations
  fixed from the reference, formatting (line breaks, `♪` music marks) preserved.

## Approach

- **Base model:** `mlx-community/gemma-4-e4b-it-4bit`, fine-tuned with **LoRA** (rank 64) via
  `mlx_lm lora` on Apple Silicon (see [config.yaml](config.yaml)).
- **Active-learning loop** ([active_learning_loop.py](active_learning_loop.py)): evaluate the
  current model → identify exact semantic failures and word mismatches → generate ~1,000 train /
  100 val pairs *targeting those specific failures* → retrain. Repeat. This concentrates data
  on what the model actually gets wrong instead of broad synthetic coverage.
- **Targeted synthetics** ([generate_targeted_synthetics.py](generate_targeted_synthetics.py)):
  pairs that drill specific gaps — line-break preservation, music notes, plural/grammar
  agreement with the reference.
- **Multilingual data** ([download_multilingual_subtitles.py](download_multilingual_subtitles.py)):
  Spanish, German, French subtitles, aligned to the source episodes.

## Layout

```text
subtitle_correction/      # package: align, parse, hallucination_filter, inference, evaluate, pipeline
active_learning_loop.py   # eval -> find failures -> generate targeted data -> retrain
generate_targeted_synthetics.py
download_multilingual_subtitles.py
config.yaml               # LoRA / training config
```

## Usage

```bash
uv run -m subtitle_correction.train          # LoRA fine-tune per config.yaml
uv run -m subtitle_correction.evaluate       # evaluate against the held-out set
uv run -m subtitle_correction.inference      # correct a subtitle file
```

## Notes

- **Training data and model checkpoints are not committed** — the corpus is derived from scraped
  subtitles (copyright), and adapters/fused weights are large. This repository is the **code and
  method**; bring your own aligned `(whisper, reference)` pairs under `data/`.
- Data-prep scripts assume a local subtitle cache; adjust the paths at the top of those scripts.
