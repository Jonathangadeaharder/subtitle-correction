import time
from pathlib import Path
from langdetect import detect, LangDetectException
from rich.console import Console

console = Console()


def load_mlx_model(model_name: str):
    """Load an MLX model + tokenizer (shared by inference and reference-free paths)."""
    from mlx_lm import load

    return load(model_name)


SYSTEM_PROMPT = (
    "You are a subtitle corrector. "
    "You receive two inputs: a Whisper transcription (correct timing, may contain mishearings or hallucinations) "
    "and a Reference subtitle (correct content, may differ in phrasing or be slightly out of sync). "
    "Output the corrected subtitle: use the Reference to fix mishearings and hallucinations in the Whisper text, "
    "but preserve the speaker's actual words and language from the Whisper where they are correct. "
    "Do NOT translate. Output only the corrected subtitle line, nothing else."
)


GERMAN_MARKERS = {
    "der",
    "die",
    "das",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    "ist",
    "nicht",
    "sind",
    "war",
    "waren",
    "wird",
    "werden",
    "wurde",
    "haben",
    "hat",
    "hatte",
    "hatten",
    "hast",
    "habt",
    "ich",
    "du",
    "sie",
    "wir",
    "ihr",
    "mich",
    "dich",
    "sich",
    "unser",
    "euer",
    "mein",
    "dein",
    "von",
    "zu",
    "auf",
    "aus",
    "bei",
    "nach",
    "vor",
    "über",
    "unter",
    "mit",
    "durch",
    "gegen",
    "ohne",
    "kann",
    "kannst",
    "muss",
    "müssen",
    "soll",
    "sollen",
    "will",
    "wollen",
    "aber",
    "oder",
    "denn",
    "weil",
    "wenn",
    "als",
    "ob",
    "dass",
    "auch",
    "noch",
    "schon",
    "immer",
    "wieder",
    "mehr",
    "sehr",
    "hier",
    "dort",
    "jetzt",
    "heute",
    "morgen",
    "gestern",
    "nein",
    "bitte",
    "danke",
    "entschuldigung",
    "gut",
    "schlecht",
    "groß",
    "klein",
    "schön",
    "Mann",
    "Frau",
    "Kind",
    "Haus",
    "Weg",
    "Tag",
    "Nacht",
    "Jahr",
    "Zeit",
    "Handy",
    "Auto",
    "Tür",
    "Fenster",
    "Tisch",
    "Stuhl",
    "Bett",
    "können",
    "dürfen",
    "möchten",
    "mögen",
    "geht",
    "mach",
    "komm",
    "geh",
    "bleib",
    "wart",
    "diese",
    "dieser",
    "dieses",
    "jene",
    "welche",
    "manche",
    "niemand",
    "etwas",
    "nichts",
    "alles",
    "jemand",
    "nur",
    "mal",
    "eben",
    "ruhig",
    "doch",
    "Beziehung",
    "Anwalt",
    "gewonnen",
    "belogen",
    "Fall",
    "beendete",
    "fing",
    "fangen",
    "willkommen",
    "Nachbarschaft",
    "Theoretisch",
    "Norden",
    "genau",
    "Kleinstadt",
    "kennen",
    "später",
    "früher",
    "werden",
    "wurden",
    "ja",
    "und",
    "nein",
    "für",
    "dem",
    "den",
    "des",
    "ein",
    "war",
    "habe",
    "hast",
    "haben",
    "hat",
    "warum",
    "wieso",
    "wie",
    "was",
    "wer",
    "wo",
    "woher",
    "wohin",
    "hier",
    "da",
    "dort",
    "jetzt",
    "gleich",
    "später",
    "wahr",
    "falsch",
    "richtig",
    "viel",
    "wenig",
    "solche",
    "welche",
    "damit",
    "darauf",
    "davon",
    "dazu",
    "daran",
    "dabei",
    "sonst",
    "zwar",
    "jedoch",
    "trotzdem",
    "oft",
    "selten",
    "manchmal",
    "stets",
    "gar",
    "sogar",
    "bereits",
    "gerne",
    "leider",
    "hoffentlich",
    "vielleicht",
    "wahrscheinlich",
    "sicher",
    "Stadt",
    "Land",
    "Welt",
    "Mensch",
    "Leben",
    "Liebe",
    "Wasser",
    "Brot",
    "Milch",
    "Wein",
    "Kaffee",
    "Hund",
    "Katze",
    "Vogel",
    "Fisch",
    "rot",
    "blau",
    "grün",
    "gelb",
    "schwarz",
    "weiß",
    "links",
    "rechts",
    "gerade",
    "oben",
    "unten",
    "Anfang",
    "Ende",
    "Hupe",
    "Bremsen",
    "Motor",
    "Reifen",
    "Stoff",
    "Farbe",
    "farb",
    "nachhaltig",
    "versuch",
    "Versuch",
    "meinem",
    "meiner",
    "deinem",
    "deiner",
    "seinem",
    "seiner",
    "Ordnung",
    "Ahnung",
    "Zustimmung",
    "vermacht",
    "Schraubstock",
    "Unbezahlbar",
    "eigentlich",
    "Seien",
    "unbesorgt",
    "keine",
    "volle",
    "schon",
    "Hallo",
    "Wunder",
    "wunder",
}


def has_german_chars(text: str) -> bool:
    return any(c in text for c in "äöüßÄÖÜ")


def has_german_patterns(text: str) -> bool:
    lower = text.lower()
    patterns = [
        r"\b(ord)nung\b",
        r"\bahnung\b",
        r"\bzustimm",
        r"\bvermacht\b",
        r"\bschraub",
        r"\bunbezahl",
        r"\bsei(en|st)\s+\w+\b",
        r"\bkeine\b",
        r"\bvolle\b",
        r"\bschaut\b",
        r"\btut\b",
        r"\bleid\b",
    ]
    import re

    return any(re.search(p, lower) for p in patterns)


def is_english(text: str, min_chars: int = 2) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    if has_german_chars(stripped):
        return False
    if has_german_patterns(stripped):
        return False
    lower = stripped.lower().rstrip(".!?,;:")
    words = [w.rstrip(".,!?;:\"'()[]{}") for w in lower.split()]
    if not words:
        return True
    german_hits = sum(1 for w in words if w in GERMAN_MARKERS)
    if german_hits >= 2:
        return False
    if german_hits == 1 and len(words) <= 4:
        return False
    if len(stripped) < 30:
        return True
    try:
        return detect(stripped) == "en"
    except LangDetectException:
        return True


class SubtitleCorrector:
    """Helper class to hold the loaded model and run corrections."""

    def __init__(
        self,
        model_name: str,
        adapter_path: Path | None = None,
        fused: bool = True,
        temp: float = 0.1,
    ):
        from mlx_lm.sample_utils import make_sampler

        self.fused = fused
        t0 = time.time()
        if fused:
            console.print(f"[dim]Loading fused model {model_name}...[/dim]")
            self.model, self.tokenizer = load_mlx_model(model_name)
        else:
            console.print(f"[dim]Loading model {model_name} with adapter {adapter_path}...[/dim]")
            from mlx_lm import load

            self.model, self.tokenizer = load(model_name, adapter_path=str(adapter_path))
        console.print(f"[dim]Model loaded in {time.time() - t0:.1f}s[/dim]")
        self.sampler = make_sampler(temp=temp)

    def correct_line(self, line: str, reference: str | None = None, max_tokens: int = 200) -> str:
        from mlx_lm import generate

        # When no reference provided, use the whisper text itself (identity prompt)
        ref = reference if reference is not None else line
        user_content = (
            f"{SYSTEM_PROMPT}\n\nWhisper transcription:\n{line}\n\nReference subtitle:\n{ref}"
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self.sampler,
            verbose=False,
        )
        text_out = response.strip()

        # Strip thinking/reasoning blocks (safety net if thinking leaks despite disable)
        for tag in ["<channel|>", "</think>", "</channel>", "<|channel|>", "<|channel>thought"]:
            if tag in text_out:
                text_out = text_out.split(tag)[-1].strip()
        if "<think>" in text_out:
            text_out = text_out.split("<think>")[-1].strip()
        # Strip raw "Thinking Process:" blocks that lack wrapper tags
        import re

        text_out = re.sub(
            r"^(?:Thinking Process|Thought|Analysis|Reasoning):\s*\n.*",
            "",
            text_out,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # Clean potential end of turn/chatml tags
        for tag in ["<|im_end|>", "<end_of_turn>", "<eos>"]:
            if tag in text_out:
                text_out = text_out.split(tag)[0]
        text_out = text_out.strip()

        # Fallback: if output is empty or wildly longer than input, return input unchanged
        if not text_out or len(text_out) > len(line) * 3 + 50:
            return line

        # Re-apply line breaks, music symbols, capitalization, and punctuation from reference
        if reference is not None:
            from .formatter import restore_formatting

            text_out = restore_formatting(text_out, reference)

        return text_out


def correct_file_impl(
    input_file: Path,
    output: Path | None = None,
    model: str = "mlx-community/gemma-4-e4b-it-4bit",
    adapter_path: Path | None = None,
    fused: bool = False,
    max_tokens: int = 800,
    temp: float = 0.1,
    lang: str = "en",
    reference_file: Path | None = None,
) -> None:

    corrector = SubtitleCorrector(model, adapter_path, fused=fused, temp=temp)
    content = input_file.read_text()

    ref_blocks = []
    if reference_file and reference_file.exists():
        ref_blocks = reference_file.read_text().strip().split("\n\n")

    if input_file.suffix == ".srt":
        blocks = content.strip().split("\n\n")
        total = len(blocks)
        corrected = []
        changed = 0
        skipped = 0
        for i, block in enumerate(blocks):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                idx = lines[0]
                timing = lines[1]
                text_lines = "\n".join(lines[2:]).strip()

                if lang and not is_english(text_lines):
                    skipped += 1
                    corrected.append(block)
                    continue

                # Fetch corresponding reference block text
                ref_text = None
                if i < len(ref_blocks):
                    ref_lines = ref_blocks[i].strip().split("\n")
                    if len(ref_lines) >= 3:
                        ref_text = "\n".join(ref_lines[2:]).strip()

                corrected_text = corrector.correct_line(
                    text_lines, reference=ref_text, max_tokens=max_tokens
                )
                if corrected_text != text_lines:
                    changed += 1
                corrected.append(f"{idx}\n{timing}\n{corrected_text}")
            else:
                corrected.append(block)
            if (i + 1) % 50 == 0:
                console.print(
                    f"[dim]  Processed {i + 1}/{total} blocks ({changed} changed, {skipped} skipped)[/dim]"
                )
        result = "\n\n".join(corrected)
        console.print(f"[green]Done: {changed}/{total} corrected, {skipped} skipped[/green]")
    else:
        ref_text = None
        if ref_blocks:
            ref_text = (
                "\n".join(ref_blocks[0].strip().split("\n")[2:])
                if reference_file.suffix == ".srt"
                else reference_file.read_text().strip()
            )
        result = corrector.correct_line(content, reference=ref_text, max_tokens=max_tokens)

    if output:
        output.write_text(result)
        console.print(f"[green]Written to {output}[/green]")
    else:
        console.print(result)
