import json
from pathlib import Path

from subtitle_correction.alignment_quality import classify_eval_entry


def test_classify_eval_entry_matches_dataset_labels() -> None:
    dataset = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.jsonl"
    if not dataset.exists():
        return
    mismatches = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            stored = entry.get("eval_scope")
            if not stored:
                continue
            if classify_eval_entry(entry) != stored:
                mismatches.append(entry["id"])
    assert mismatches == []