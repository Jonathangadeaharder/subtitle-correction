"""Deterministic reference-based subtitle correction (no LLM).

Given the exact reference narration prose (the text fed to TTS) and a Whisper
SRT (correct timing, corrupted text), this aligns the reference token stream
onto the concatenated cue texts via ``difflib.SequenceMatcher`` and slices the
reference back into per-cue spans. Every cue keeps its original index and
timestamp; only the text is replaced, verbatim, from the reference.

Invariants enforced by the asserts in ``align_reference_to_srt``:
- cue count and every timestamp line are preserved exactly;
- each corrected cue's text is a contiguous token-span of the reference;
- no reference token is dropped (the concatenation of all cue texts equals the
  whitespace-joined reference tokens).
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

import pysrt

_TOKEN_RE = re.compile(r"\S+")
_NORM_RE = re.compile(r"[^0-9a-zäöüáéíóúàèïü]")


def _default_gretel_root() -> Path:
    return Path(os.getenv("GRETEL_ROOT", Path(__file__).resolve().parents[2] / "Gretel"))


def _tokens(text: str) -> list[str]:
    """Whitespace-delimited tokens (punctuation stays attached to the word)."""
    return _TOKEN_RE.findall(text)


def _norm(token: str) -> str:
    """Lowercase, strip punctuation/diacritic noise so misheard pairs still align."""
    return _NORM_RE.sub("", token.lower())


def strip_markdown_prose(text: str) -> str:
    """Drop markdown heading lines; keep the narration body."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return body.strip()


def _assign_reference_to_cues(
    cue_token_owner: list[int],
    whisper_norm: list[str],
    ref_norm: list[str],
    n_cues: int,
) -> list[int]:
    """Return ref_to_cue[j] = index of the cue that owns reference token j.

    Matched reference tokens inherit the cue of their aligned whisper token.
    Replacement tokens inherit the cue of the whisper token(s) they replace.
    Inserted reference-only gaps carry forward to the previous token's cue, so
    interior/trailing reference text that Whisper dropped is never lost.
    """
    sm = difflib.SequenceMatcher(a=whisper_norm, b=ref_norm, autojunk=False)
    filled: list[int | None] = [None] * len(ref_norm)
    last = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                owner = cue_token_owner[i1 + offset]
                filled[j1 + offset] = owner
                last = owner
        elif tag == "replace" and i2 > i1:
            source_len = i2 - i1
            target_len = j2 - j1
            for offset in range(target_len):
                owner_offset = min(source_len - 1, (offset * source_len) // target_len)
                owner = cue_token_owner[i1 + owner_offset]
                filled[j1 + offset] = owner
                last = owner
        elif tag == "insert":
            for j in range(j1, j2):
                filled[j] = last
        # delete has no reference tokens to assign.

    assigned = [last if owner is None else owner for owner in filled]
    # Defensive: monotonic non-decreasing (alignment blocks are already ordered).
    for i in range(1, len(assigned)):
        if assigned[i] < assigned[i - 1]:
            assigned[i] = assigned[i - 1]
    assert len(assigned) == len(ref_norm)
    assert not assigned or 0 <= min(assigned) and max(assigned) < n_cues
    return assigned


def align_reference_to_srt(
    reference_text: str,
    whisper_srt: Path | str,
    out: Path | str,
) -> Path:
    """Align reference prose onto a Whisper SRT and write a corrected SRT.

    ``reference_text`` may include a markdown heading; it is stripped.
    Returns the output path. Raises AssertionError if any invariant is violated.
    """
    out = Path(out)
    subs = pysrt.open(str(whisper_srt))
    n_cues = len(subs)

    ref_body = strip_markdown_prose(reference_text)
    ref_tokens = _tokens(ref_body)

    # Concatenate cue texts, remembering which cue each whisper token came from.
    cue_token_owner: list[int] = []
    whisper_tokens: list[str] = []
    for ci, sub in enumerate(subs):
        for tok in _tokens(sub.text):
            whisper_tokens.append(tok)
            cue_token_owner.append(ci)

    ref_to_cue = _assign_reference_to_cues(
        cue_token_owner,
        [_norm(t) for t in whisper_tokens],
        [_norm(t) for t in ref_tokens],
        n_cues,
    )

    # Slice the reference token stream into per-cue spans.
    cue_tokens: list[list[str]] = [[] for _ in range(n_cues)]
    for tok, ci in zip(ref_tokens, ref_to_cue):
        cue_tokens[ci].append(tok)
    corrected_texts = [" ".join(toks) for toks in cue_tokens]

    # --- Invariants ---------------------------------------------------------
    # 1. No reference token dropped: concatenation round-trips the reference.
    assert _tokens(" ".join(corrected_texts)) == ref_tokens, (
        "reference tokens were dropped or reordered during slicing"
    )
    # 2. Every corrected cue text is a contiguous token-span of the reference.
    cursor = 0
    for text in corrected_texts:
        span = _tokens(text)
        assert ref_tokens[cursor : cursor + len(span)] == span, (
            "corrected cue text is not a verbatim reference span"
        )
        cursor += len(span)
    assert cursor == len(ref_tokens)

    write_preserving_timing(Path(str(whisper_srt)), out, corrected_texts)

    # 3. Post-write: cue count + timestamps identical to the source SRT.
    _assert_timing_preserved(Path(str(whisper_srt)), out)
    return out


def write_preserving_timing(src_srt: Path, out: Path, new_texts: list[str]) -> None:
    """Rebuild the SRT replacing ONLY each block's text payload.

    Index and timing lines are copied verbatim from the source so they remain
    byte-identical; only the lines after the timing line are swapped.
    """
    raw = src_srt.read_text(encoding="utf-8")
    blocks = re.split(r"\r?\n\r?\n", raw.lstrip("﻿").strip())
    if len(blocks) != len(new_texts):
        raise ValueError(
            f"block count {len(blocks)} != corrected count {len(new_texts)}; refusing to write"
        )
    rebuilt = []
    for block, new_text in zip(blocks, new_texts):
        lines = block.split("\n")
        # lines[0] = index, lines[1] = timing, lines[2:] = original (discarded) text.
        rebuilt.append("\n".join(lines[:2]) + "\n" + new_text)
    out.write_text("\n\n".join(rebuilt) + "\n", encoding="utf-8")


def _assert_timing_preserved(src_srt: Path, out_srt: Path) -> None:
    a = pysrt.open(str(src_srt))
    b = pysrt.open(str(out_srt))
    assert len(a) == len(b), f"cue count changed: {len(a)} -> {len(b)}"
    for o, c in zip(a, b):
        assert o.index == c.index, f"index changed: {o.index} -> {c.index}"
        o_ts, c_ts = f"{o.start} --> {o.end}", f"{c.start} --> {c.end}"
        assert o_ts == c_ts, f"timestamp changed at cue {o.index}: {o_ts!r} -> {c_ts!r}"


def unified_diff(original_srt: Path | str, corrected_srt: Path | str) -> str:
    a = Path(str(original_srt)).read_text(encoding="utf-8").splitlines(keepends=True)
    b = Path(str(corrected_srt)).read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a, b, fromfile=str(original_srt), tofile=str(corrected_srt))
    )


def _self_check() -> None:
    """Synthetic check of the gap carry-forward: dropped + misheard reference text."""
    # 3 cues. Whisper "b" is misheard ("bee"); reference inserts a dropped "extra gap".
    cue_owner = [0, 0, 1, 1, 2]  # cue0="a bee", cue1="see dee", cue2="ee"
    whisper_norm = ["a", "bee", "see", "dee", "ee"]
    ref_norm = ["a", "b", "see", "dee", "extra", "gap", "ee"]
    assigned = _assign_reference_to_cues(cue_owner, whisper_norm, ref_norm, n_cues=3)
    # "a"->0; "b"(misheard gap)->carry 0; "see","dee"->1; "extra","gap"(dropped)->carry 1; "ee"->2
    assert assigned == [0, 0, 1, 1, 1, 1, 2], assigned
    assert assigned == sorted(assigned)  # monotonic
    replacement = _assign_reference_to_cues(
        cue_token_owner=[0, 1],
        whisper_norm=["gutes", "maus"],
        ref_norm=["gutes", "haus"],
        n_cues=2,
    )
    assert replacement == [0, 1], replacement
    print("self-check OK: gap carry-forward assigns every reference token monotonically.")


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Reference-based SRT correction (deterministic, no LLM)."
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="run the synthetic alignment check and exit",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=_default_gretel_root() / "outputs/narration/ch01_der_endlose_regen.md",
    )
    p.add_argument(
        "--whisper-srt",
        type=Path,
        default=_default_gretel_root() / "shot_production/chapter1/outputs/ch1_transcript.srt",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_default_gretel_root()
        / "shot_production/chapter1/outputs/ch1_transcript.corrected.srt",
    )
    args = p.parse_args()

    if args.self_check:
        _self_check()
        return

    out = align_reference_to_srt(
        args.reference.read_text(encoding="utf-8"), args.whisper_srt, args.out
    )
    print(f"Wrote {out}")
    print(
        "Invariants OK: cue count + timestamps preserved, every cue is a verbatim reference span.\n"
    )
    print(unified_diff(args.whisper_srt, out))


if __name__ == "__main__":
    _main()
