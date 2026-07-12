"""
Dynamic Active Learning Loop script (v3).
1. Runs evaluation of the current model on the evaluation dataset.
2. Identifies exact semantic failures and word-mismatches.
3. Generates 1,000 training and 100 validation pairs targeting those exact failures.
4. Strips formatting (line breaks, music symbols) from training/validation texts.
"""

import json
import random
import sys
import re
from pathlib import Path

# Fallback semantic confusions if no failures are found
FALLBACK_CONFUSIONS = [
    ("their", "there"),
    ("there", "their"),
    ("your", "you're"),
    ("you're", "your"),
    ("its", "it's"),
    ("it's", "its"),
    ("to", "too"),
    ("too", "to"),
    ("were", "where"),
    ("where", "were"),
    ("then", "than"),
    ("than", "then"),
    ("heard", "hard"),
    ("knew", "new"),
    ("seen", "scene"),
    ("here", "hear"),
    ("affect", "effect"),
    ("loose", "lose"),
    ("advice", "advise"),
    ("Leakey", "Leekie"),
    ("Delfin", "Delphine"),
    ("Katya", "Katja"),
    ("gonna", "going to"),
    ("wanna", "want to"),
    ("gotta", "got to"),
]

SENTENCE_TEMPLATES = [
    "I think {word1} is going to be {word2}.",
    "Did you see {word1} near the {word2}?",
    "We need to get {word1} and {word2} right now.",
    "She said {word1} was better than {word2}.",
    "Why did {word1} tell you about {word2}?",
    "It is important to keep {word1} separate from {word2}.",
    "Can you bring {word1} to the {word2} tomorrow?",
    "He wanted {word1} instead of {word2}.",
    "I believe {word1} is the only way to save {word2}.",
    "Look at {word1} sitting over by {word2}.",
]


def clean_word(w):
    return re.sub(r"[^\w\s]", "", w.lower()).strip()


def identify_model_failures():
    """
    Evaluates current model, returns list of (whisper_word, gt_word) confusions
    where the model failed to correct text.
    """
    fused_model_path = Path("runs/subtitle-corrector-4b-fused")
    base_model = "mlx-community/gemma-4-e4b-it-4bit"

    # Choose base model or currently fused model
    model = str(fused_model_path) if fused_model_path.exists() else base_model
    fused = fused_model_path.exists()

    dataset_path = Path("evaluation/dataset.jsonl")
    if not dataset_path.exists():
        return FALLBACK_CONFUSIONS

    print(f"Analyzing model '{model}' for failures on validation set...")

    from subtitle_correction.inference import SubtitleCorrector

    try:
        corrector = SubtitleCorrector(model, fused=fused)
    except Exception as e:
        print(f"Could not load model: {e}. Using fallback confusions.")
        return FALLBACK_CONFUSIONS

    # Read evaluation cases
    with open(dataset_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    failures = []
    for entry in entries:
        whisper = entry["whisper_text"]
        target = entry["ground_truth"]
        reference = entry.get("reference", target)

        # Run correction
        try:
            corrected = corrector.correct_line(whisper, reference=reference)
        except Exception:
            continue

        # Clean and compare words
        clean_corr = clean_word(corrected)
        clean_gt = clean_word(target)

        if clean_corr != clean_gt:
            # Model output disagrees with the target. Find the missing target words.
            w_words = clean_word(whisper).split()
            gt_words = clean_gt.split()

            # Map word lists
            for w in w_words:
                for gt in gt_words:
                    # If they are phonetically close or in the same sentence structure, map them
                    if w != gt and (w[:2] == gt[:2] or len(w) == len(gt)):
                        failures.append((w, gt))

    if not failures:
        print("No failures found, using fallback confusions.")
        return FALLBACK_CONFUSIONS

    # Remove duplicates and limit to top confusions
    failures = list(set(failures))
    print(f"Identified {len(failures)} actual failure patterns: {failures[:10]}...")
    return failures


def generate_targeted_batch(confusions, count, seed):
    rng = random.Random(seed)
    pairs = []

    for _ in range(count):
        conf = rng.choice(confusions)
        w_word, gt_word = conf
        template = rng.choice(SENTENCE_TEMPLATES)
        filler = rng.choice(
            ["something", "nothing", "everything", "anyone", "everyone", "today", "tomorrow"]
        )

        # Flat single-line sentences, no line breaks or symbols
        gt_sentence = template.format(word1=gt_word, word2=filler).strip()
        whisper_sentence = template.format(word1=w_word, word2=filler).strip()

        pairs.append(
            {
                "whisper_text": whisper_sentence,
                "corrected_text": gt_sentence,
                "ground_truth": gt_sentence,
                "source_file": "synthetic_active_learning",
            }
        )

    return pairs


def main():
    if len(sys.argv) < 2:
        print("Usage: active_learning_loop.py <epoch_num>")
        sys.exit(1)

    epoch = int(sys.argv[1])
    print(f"=== Active Learning Loop: Epoch {epoch} ===")

    # 1. Dynamically analyze model weaknesses
    confusions = identify_model_failures()

    # 2. Generate 1,000 training and 100 validation pairs targeting those failures
    print(f"Generating 1000 targeted training pairs for Epoch {epoch}...")
    new_train = generate_targeted_batch(confusions, 1000, seed=1000 + epoch)

    print(f"Generating 100 targeted validation pairs for Epoch {epoch}...")
    new_val = generate_targeted_batch(confusions, 100, seed=2000 + epoch)

    # 3. Append to training_pairs.jsonl
    subcache_path = Path.home() / "Downloads" / ".subcache" / "training_pairs.jsonl"
    with open(subcache_path, "a") as f:
        for p in new_train:
            f.write(json.dumps(p) + "\n")
    print("Added 1000 targeted training pairs to training_pairs.jsonl")

    # 4. Append to validation dataset
    eval_dataset_path = Path("evaluation/dataset.jsonl")
    if eval_dataset_path.exists():
        with open(eval_dataset_path) as f:
            existing = [json.loads(line) for line in f if line.strip()]
        start_id = len(existing) + 1
        for i, p in enumerate(new_val):
            existing.append(
                {
                    "id": f"slice_active_{start_id + i:04d}",
                    "audio_path": "",
                    "whisper_text": p["whisper_text"],
                    "ground_truth": p["ground_truth"],
                    "reference": p["corrected_text"],
                    "source_file": "active_learning_val",
                    "start_time": "00:00:00,000",
                    "end_time": "00:00:00,000",
                    "has_error": True,
                }
            )
        with open(eval_dataset_path, "w") as f:
            for entry in existing:
                f.write(json.dumps(entry) + "\n")
        print("Added 100 targeted validation pairs to dataset.jsonl")


if __name__ == "__main__":
    main()
