"""Context-window subtitle correction without a per-cue reference (German Whisper)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pysrt
from rich.console import Console

from .srt_from_reference import unified_diff, write_preserving_timing

__all__ = ["unified_diff", "write_preserving_timing"]

console = Console()

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
CONTEXT_WINDOW = 2

SYSTEM_PROMPT = (
    "Du korrigierst deutsche Untertitel aus einer automatischen Spracherkennung (Whisper). "
    "Die Erkennung enthaelt Verhoerer (falsch verstandene, aehnlich klingende Woerter), "
    "falsche Gross-/Kleinschreibung mitten im Satz, fehlende Anfuehrungszeichen bei "
    "woertlicher Rede und ueberfluessige Kommata.\n"
    "Du bekommst den vorherigen und den naechsten Untertitel als Kontext, um Verhoerer "
    "im Satzzusammenhang aufzuloesen. Korrigiere AUSSCHLIESSLICH den mit [AKTUELL] "
    "markierten Untertitel.\n"
    "Regeln:\n"
    "- Gib NUR den korrigierten Text des aktuellen Untertitels zurueck, ohne Marker, "
    "ohne Erklaerung.\n"
    "- Aendere ein Wort NUR, um einen Verhoerer, einen Tippfehler, falsche "
    "Gross-/Kleinschreibung oder Zeichensetzung zu korrigieren. Ersetze NIEMALS ein "
    "Wort durch ein Synonym und formuliere den Satz nicht um.\n"
    "- Ist ein Wort bereits korrektes Deutsch und passt in den Satz, lass es exakt "
    "unveraendert.\n"
    "- Erfinde keine Woerter. Korrigiere einen Verhoerer nur zu einem existierenden "
    "deutschen Wort, das aehnlich klingt; bist du dir nicht sicher, lass das Wort "
    "unveraendert.\n"
    "- Fuege keine Woerter oder Saetze hinzu und veraendere die Zeilenumbrueche nicht.\n"
    "- Korrigiere Rechtschreibung, Gross-/Kleinschreibung und Zeichensetzung nach "
    "deutscher Norm.\n"
    "- Uebersetze nicht. Die Sprache bleibt Deutsch."
)


def build_prompt(prev_texts: list[str], cur_text: str, next_texts: list[str]) -> str:
    prev_block = "\n".join(prev_texts) if prev_texts else "(kein vorheriger Untertitel)"
    next_block = "\n".join(next_texts) if next_texts else "(kein naechster Untertitel)"
    return (
        f"[VORHER]\n{prev_block}\n\n[AKTUELL]\n{cur_text}\n\n[NACHHER]\n{next_block}"
        "\n\nKorrigierter Text des aktuellen Untertitels:"
    )


def _strip_model_noise(text: str, original: str) -> str:
    if "<|channel>final" in text:
        text = text.split("<|channel>final")[-1]
    for tag in ("</think>", "<|channel>thought", "<|think|>", "<think>"):
        if tag in text:
            text = text.split(tag)[-1]
    text = re.sub(r"^\s*<\|message\|?>\s*", "", text)
    for tag in (
        "<|im_end|>",
        "<end_of_turn>",
        "<eos>",
        "<turn|>",
        "<start_of_turn>",
        "<|turn>",
    ):
        if tag in text:
            text = text.split(tag)[0]
    text = re.sub(
        r"^\s*\[(?:AKTUELL|VORHER|NACHHER)\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:Korrigierter Text[^\\n:]*:|Korrektur:|Antwort:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if (
        len(text) >= 2
        and text[0] in "\"'"
        and text[-1] == text[0]
        and original.strip()[:1] not in "\"'"
    ):
        text = text[1:-1].strip()
    return text.strip()


class ContextCorrector:
    def __init__(self, model_name: str = BASE_MODEL, temp: float = 0.0):
        from mlx_lm.sample_utils import make_sampler

        from .inference import load_mlx_model

        console.print(f"[dim]Loading base instruct model {model_name}...[/dim]")
        self.model, self.tokenizer = load_mlx_model(model_name)
        self.sampler = make_sampler(temp=temp)

    def correct(
        self,
        prev_texts: list[str],
        cur_text: str,
        next_texts: list[str],
        *,
        max_tokens: int = 200,
    ) -> str:
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(prev_texts, cur_text, next_texts),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self.sampler,
            verbose=False,
        )
        out = _strip_model_noise(raw.strip(), cur_text)
        if not out or len(out) > len(cur_text) * 3 + 40:
            return cur_text
        return out


def correct_srt(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str = BASE_MODEL,
    temp: float = 0.0,
) -> tuple[int, int]:
    subs = pysrt.open(str(input_path))
    texts = [s.text for s in subs]
    corrector = ContextCorrector(model_name=model_name, temp=temp)
    corrected_texts: list[str] = []
    changed = 0
    for i, text in enumerate(texts):
        prev = texts[max(0, i - CONTEXT_WINDOW) : i]
        nxt = texts[i + 1 : i + 1 + CONTEXT_WINDOW]
        out = corrector.correct(prev, text, nxt)
        if out != text:
            changed += 1
        corrected_texts.append(out)
    write_preserving_timing(input_path, output_path, corrected_texts)
    return changed, len(texts)


def self_check() -> None:
    out = _strip_model_noise("<think>noise</think>Korrektur: Hallo", "Helo")
    assert out == "Hallo", out
    prompt = build_prompt(["a"], "b", ["c"])
    assert "[AKTUELL]" in prompt and "b" in prompt
    print("self-check OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--input", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --self-check")
    changed, total = correct_srt(args.input, args.output, model_name=args.model, temp=args.temp)
    print(f"Wrote {args.output} ({changed}/{total} cues changed)")


if __name__ == "__main__":
    main()
