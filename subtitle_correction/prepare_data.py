import json
import random
from pathlib import Path

import typer
from rich.console import Console

SYSTEM_PROMPT = (
    "You are a subtitle corrector. "
    "You receive two inputs: a Whisper transcription (correct timing, may contain mishearings or hallucinations) "
    "and a Reference subtitle (correct content, may differ in phrasing or be slightly out of sync). "
    "Output the corrected subtitle: use the Reference to fix mishearings and hallucinations in the Whisper text, "
    "but preserve the speaker's actual words and language from the Whisper where they are correct. "
    "Do NOT translate. Output only the corrected subtitle line, nothing else."
)

console = Console()


def format_chat_text(tokenizer, system: str, whisper: str, reference: str, assistant: str) -> str:
    # Strip music notes and line breaks from training inputs/outputs
    def clean(s):
        return s.replace("♪", "").replace("\n", " ").replace("\r", "").strip()

    user_content = (
        f"{system}\n\n"
        f"Whisper transcription:\n{clean(whisper)}\n\n"
        f"Reference subtitle:\n{clean(reference)}"
    )
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": clean(assistant)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)


def align_casing_and_punctuation(source: str, target: str) -> str:
    """Ensure source has identical casing on the first word and matches trailing punctuation of target."""
    s_words = source.strip().split()
    t_words = target.strip().split()
    if not s_words or not t_words:
        return source

    # Match capitalization of first word
    if t_words[0] and t_words[0][0].isupper():
        s_words[0] = s_words[0][0].upper() + s_words[0][1:] if s_words[0] else ""
    elif t_words[0] and t_words[0][0].islower():
        s_words[0] = s_words[0][0].lower() + s_words[0][1:] if s_words[0] else ""

    # Remove trailing punctuation from source
    last_word = s_words[-1]
    while last_word and last_word[-1] in ".,!?\"';:♪":
        last_word = last_word[:-1]

    # Extract trailing punctuation from target
    target_punc = ""
    t_last = t_words[-1]
    while t_last and t_last[-1] in ".,!?\"';:♪":
        target_punc = t_last[-1] + target_punc
        t_last = t_last[:-1]

    s_words[-1] = last_word + target_punc
    return " ".join(s_words)


HOMOPHONES = {
    "their": ["there", "they're"],
    "there": ["their", "they're"],
    "they're": ["their", "there"],
    "your": ["you're"],
    "you're": ["your"],
    "its": ["it's"],
    "it's": ["its"],
    "to": ["too", "two"],
    "too": ["to", "two"],
    "two": ["to", "too"],
    "were": ["where", "we're"],
    "where": ["were", "we're"],
    "we're": ["were", "where"],
    "whose": ["who's"],
    "who's": ["whose"],
    "then": ["than"],
    "than": ["then"],
    "accept": ["except"],
    "except": ["accept"],
    "affect": ["effect"],
    "effect": ["affect"],
    "heard": ["hard"],
    "hard": ["heard"],
    "through": ["threw"],
    "threw": ["through"],
    "right": ["write", "rite"],
    "write": ["right", "rite"],
    "know": ["no"],
    "no": ["know"],
    "new": ["knew"],
    "knew": ["new"],
    "piece": ["peace"],
    "peace": ["piece"],
    "scene": ["seen"],
    "seen": ["scene"],
    "week": ["weak"],
    "weak": ["week"],
    "weather": ["whether"],
    "whether": ["weather"],
    "fair": ["fare"],
    "fare": ["fair"],
    "meat": ["meet"],
    "meet": ["meat"],
    "plain": ["plane"],
    "plane": ["plain"],
    "board": ["bored"],
    "bored": ["board"],
    "course": ["coarse"],
    "coarse": ["course"],
    "hear": ["here"],
    "here": ["hear"],
    "hole": ["whole"],
    "whole": ["hole"],
    "morning": ["mourning"],
    "mourning": ["morning"],
    "principal": ["principle"],
    "principle": ["principal"],
    "stationary": ["stationery"],
    "stationery": ["stationary"],
    "wait": ["weight"],
    "weight": ["wait"],
    "ware": ["wear", "where"],
    "wear": ["ware"],
}

CONTRACTIONS = {
    "going to": ["gonna", "going to"],
    "want to": ["wanna", "want to"],
    "got to": ["gotta", "got to"],
    "have to": ["hafta"],
    "kind of": ["kinda"],
    "sort of": ["sorta"],
    "out of": ["outta"],
    "because": ["cuz", "cause"],
    "don't you": ["dontcha"],
    "give me": ["gimme"],
    "let me": ["lemme"],
    "get out": ["getout"],
}

PHONETIC_SUBS = {
    "probably": ["probly", "prolly"],
    "something": ["somethin", "sumpin"],
    "nothing": ["nothin", "nuthin", "no thing"],
    "everything": ["everythin", "every thing"],
    "anything": ["anythin"],
    "remember": ["member", "rember"],
    "because": ["cuz", "cause"],
    "actually": ["actualy", "actuly"],
    "definitely": ["definitly", "definately"],
    "separate": ["seperate"],
    "tomorrow": ["tomarrow", "tommorow"],
    "beautiful": ["beautifull", "butiful"],
    "different": ["differnt", "diffrent"],
    "interesting": ["intresting"],
    "neighbor": ["nieghbor"],
    "people": ["peeple"],
    "police": ["pleese"],
    "someone": ["some one"],
    "everyone": ["every one"],
    "already": ["all ready"],
    "alright": ["all right"],
    "cannot": ["can not"],
    "maybe": ["may be"],
    "away": ["a way"],
    "sometimes": ["some times"],
    "anyone": ["any one"],
    "instead": ["in stead"],
    "without": ["with out"],
    "myself": ["my self"],
    "yourself": ["your self"],
    "tonight": ["to night"],
    "today": ["to day"],
    "somehow": ["some how"],
    "whatever": ["what ever"],
    "however": ["how ever"],
    "overall": ["over all"],
    "anymore": ["any more"],
    "anyway": ["any way"],
}

SIMILAR_WORDS = {
    "he": ["the", "she"],
    "she": ["the", "he"],
    "his": ["this", "her"],
    "her": ["the", "his"],
    "this": ["his", "the"],
    "and": ["in", "an"],
    "was": ["as", "what"],
    "what": ["was"],
    "with": ["which"],
    "which": ["with"],
    "for": ["from", "four"],
    "from": ["for"],
    "that": ["what", "at"],
    "not": ["now", "no"],
    "now": ["not", "no"],
    "been": ["being", "in"],
    "has": ["his", "as"],
    "had": ["has", "at"],
    "but": ["what"],
    "all": ["old"],
    "one": ["won", "on"],
    "when": ["went", "then"],
    "then": ["when", "than"],
    "them": ["then", "him"],
    "him": ["them"],
    "like": ["look"],
    "look": ["like"],
    "back": ["black", "bark"],
    "just": ["must", "jest"],
    "time": ["dime", "tie"],
    "some": ["come", "son"],
    "come": ["some"],
    "here": ["her", "hear"],
    "there": ["their", "the"],
}


def _corrupt_text(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text

    # Apply double words (Whisper duplication artifact)
    if rng.random() < 0.15:
        idx = rng.randint(0, len(words) - 1)
        words.insert(idx, words[idx])

    # Apply leading conjunction additions
    if rng.random() < 0.15:
        words.insert(0, rng.choice(["And", "But", "So", "and", "but", "so"]))

    # Apply normal token corruptions
    num_corruptions = rng.randint(1, max(1, len(words) // 3))
    for _ in range(num_corruptions):
        if not words:
            break
        idx = rng.randint(0, len(words) - 1)
        word_lower = words[idx].lower().strip(".,!?;:\"'")
        punctuation = ""
        stripped = words[idx]
        while stripped and stripped[-1] in ".,!?;:\"'":
            punctuation = stripped[-1] + punctuation
            stripped = stripped[:-1]
        word_lower = stripped.lower()

        corruption_type = rng.choice(
            ["homophone", "contraction", "phonetic", "similar", "repeat", "drop", "truncate"]
        )

        if corruption_type == "homophone" and word_lower in HOMOPHONES:
            replacement = rng.choice(HOMOPHONES[word_lower])
            words[idx] = replacement + punctuation
        elif corruption_type == "contraction":
            bigram = " ".join(words[idx : idx + 2]).lower().strip(".,!?;:\"'")
            if bigram in CONTRACTIONS:
                replacement = rng.choice(CONTRACTIONS[bigram])
                words[idx] = replacement + punctuation
                if idx + 1 < len(words):
                    words.pop(idx + 1)
        elif corruption_type == "phonetic" and word_lower in PHONETIC_SUBS:
            replacement = rng.choice(PHONETIC_SUBS[word_lower])
            words[idx] = replacement + punctuation
        elif corruption_type == "similar" and word_lower in SIMILAR_WORDS:
            replacement = rng.choice(SIMILAR_WORDS[word_lower])
            preserve_case = words[idx][0].isupper()
            if preserve_case:
                replacement = replacement.capitalize()
            words[idx] = replacement + punctuation
        elif corruption_type == "repeat" and len(word_lower) > 2:
            repeat_count = rng.randint(2, 4)
            words[idx] = (stripped + " ") * repeat_count + stripped + punctuation
        elif corruption_type == "drop" and word_lower in {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "of",
            "and",
            "or",
        }:
            words.pop(idx)
        elif corruption_type == "truncate" and len(word_lower) > 4:
            trunc_len = rng.randint(len(word_lower) - 2, len(word_lower) - 1)
            words[idx] = stripped[:trunc_len] + punctuation

    return " ".join(words)


def prepare_data_impl(
    input_file: Path,
    output_dir: Path = Path("data"),
    val_split: float = 0.1,
    seed: int = 42,
    augment: int = 3,
    identity_ratio: float = 0.25,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    pairs = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    if not pairs:
        console.print("[red]No training pairs found[/red]")
        raise typer.Exit(code=1)

    console.print(f"Loaded {len(pairs)} raw pairs")

    valid_pairs = [
        p
        for p in pairs
        if p.get("whisper_text", "").strip() and p.get("corrected_text", "").strip()
    ]
    console.print(f"Valid pairs for training: {len(valid_pairs)}")

    import yaml
    from transformers import AutoTokenizer

    config_path = Path(__file__).parent.parent / "config.yaml"
    model_name = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            model_name = cfg.get("model", model_name)

    console.print(f"Loading tokenizer for model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    correct_texts = set()
    examples = []

    for pair in valid_pairs:
        whisper_text = pair.get("whisper_text", "").strip()
        # corrected_text is the aligned OpenSubtitles line — it IS the reference AND the ground truth
        corrected_text = pair.get("corrected_text", "").strip()
        # ground_truth may be manually set; falls back to corrected_text
        ground_truth = pair.get("ground_truth", corrected_text).strip()
        if not whisper_text or not corrected_text:
            continue

        text = format_chat_text(
            tokenizer, SYSTEM_PROMPT, whisper_text, corrected_text, ground_truth
        )
        examples.append({"text": text})
        correct_texts.add(corrected_text)

    console.print(f"Real pairs (whisper + reference -> ground truth): {len(examples)}")

    augmented = 0
    for correct_text in correct_texts:
        for _ in range(augment):
            corrupted = _corrupt_text(correct_text, rng)
            if corrupted != correct_text:
                ref = correct_text
                out = correct_text
                # Programmatically inject newlines or music notes to teach preservation
                if rng.random() < 0.30 and " " in ref and "\n" not in ref:
                    words = ref.split()
                    split_idx = len(words) // 2
                    ref = " ".join(words[:split_idx]) + "\n" + " ".join(words[split_idx:])
                    out = ref
                    # Match spacing in corrupted input
                    c_words = corrupted.split()
                    c_split = min(split_idx, len(c_words))
                    corrupted = " ".join(c_words[:c_split]) + "\n" + " ".join(c_words[c_split:])

                if rng.random() < 0.30 and "♪" not in ref:
                    ref = f"♪ {ref} ♪"
                    out = ref
                    corrupted = f"♪ {corrupted} ♪"

                text = format_chat_text(tokenizer, SYSTEM_PROMPT, corrupted, ref, out)
                examples.append({"text": text})
                augmented += 1

    console.print(f"Synthetic mishearing pairs: {augmented}")

    identity_count = int(len(correct_texts) * identity_ratio)
    for correct_text in list(correct_texts)[:identity_count]:
        ref = correct_text
        out = correct_text
        if rng.random() < 0.30 and " " in ref and "\n" not in ref:
            words = ref.split()
            split_idx = len(words) // 2
            ref = " ".join(words[:split_idx]) + "\n" + " ".join(words[split_idx:])
            out = ref
        if rng.random() < 0.30 and "♪" not in ref:
            ref = f"♪ {ref} ♪"
            out = ref

        text = format_chat_text(tokenizer, SYSTEM_PROMPT, ref, ref, out)
        examples.append({"text": text})
    console.print(f"Identity (stability) pairs: {identity_count}")

    console.print(f"[bold]Total examples: {len(examples)}[/bold]")

    rng.shuffle(examples)

    val_count = max(1, int(len(examples) * val_split))
    train_examples = examples[val_count:]
    val_examples = examples[:val_count]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    console.print(f"[green]Train: {len(train_examples)} -> {train_path}[/green]")
    console.print(f"[green]Val: {len(val_examples)} -> {val_path}[/green]")
