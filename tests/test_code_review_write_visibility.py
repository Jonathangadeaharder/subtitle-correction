"""Enforce visibility of API writes in the code-review workflow.

On PR #12 (run 35545918812) the resolve step's write burst tripped
GitHub's secondary rate limit: every review-finding POST failed, every
failure was swallowed by 2>/dev/null / 2>&1 / || true, and the gate
blocked a PR on findings nobody could see (issue #15). These tests pin
the failure modes the fix must keep closed:

- no write call may redirect its stderr or discard its exit code;
- the resolve step must pace its writes and report what it did;
- a round that posts zero inline comments must be loud, and the fallback
  PR comment is the last visibility path: its failure fails the step.
"""

import re

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "code-review.yml"

# gh api calls that perform writes. Reads may redirect stderr; writes
# must not, or a rate-limited POST looks like success.
WRITE_LINE = re.compile(
    r"gh api .*(--method POST|-f body=|graphql -f query).*"
)
SWALLOW = re.compile(r"(2>/dev/null|2>&1|\|\| true)")


def logical_lines(text: str) -> list[str]:
    # Write calls span backslash-continued lines in the YAML run blocks;
    # join them so one gh api invocation is one scannable line.
    return text.replace("\\\n", " ").splitlines()


RESOLVE_STEP = "Resolve previous review comments"
POST_STEP = "Post review comments"


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def step_run_text(name: str) -> str:
    workflow = yaml.safe_load(workflow_text())
    steps = workflow["jobs"]["ocr-review"]["steps"]
    matching = [s for s in steps if s.get("name") == name]
    assert matching, f"step {name!r} not found in code-review.yml"
    return matching[0]["run"]


def test_no_write_call_swallows_its_errors() -> None:
    offenders = [
        line.strip()
        for line in logical_lines(workflow_text())
        if WRITE_LINE.search(line) and SWALLOW.search(line)
    ]
    assert not offenders, (
        "API write calls in code-review.yml must not redirect stderr or "
        "discard their exit code. On PR #12 this made every finding POST "
        f"fail invisibly. Offending lines: {offenders}"
    )


def test_resolve_step_paces_and_counts_its_writes() -> None:
    run = step_run_text(RESOLVE_STEP)
    # A ~110-write burst in ~30s is what tripped the secondary rate limit
    # on PR #12, and a fixed sleep only measures throttling. Writes must
    # retry with backoff so transient throttling is overcome, and the
    # outcome of every write must be counted.
    assert "gh_with_backoff" in run, (
        "the resolve step must retry its writes with exponential backoff, "
        "not fire blind (issue #15: the unthrottled burst rate-limited the "
        "whole run)"
    )
    assert "gh-with-backoff.sh" in run, (
        "the retry helper lives in scripts/gh-with-backoff.sh; sourcing it "
        "beats duplicating it verbatim in both steps"
    )
    assert "gh_read_with_backoff" in run, (
        "the comments fetch must retry too: a throttled read of previous "
        "comments otherwise resolves nothing for the whole round"
    )
    assert re.search(r"replied=\$", run) and re.search(r"minimized=\$", run), (
        "the resolve step must count and surface replied/minimized/failed "
        "in its step output, not run silently"
    )
    # A failed fetch of previous comments must be distinguishable from
    # "no previous comments" in the log.
    assert "exit 0" in run and "Could not fetch" in run, (
        "a failed comments fetch must warn and be visible, not read as "
        "'no previous comments'"
    )


def test_post_step_fails_loudly_when_nothing_was_posted() -> None:
    run = step_run_text(POST_STEP)
    assert re.search(r"posted=\$POSTED", run), (
        "the post step must echo posted/failed counts into the log"
    )
    assert "gh_with_backoff" in run, (
        "inline comment posts must retry with backoff: a burst can trip the "
        "same secondary rate limit that killed every POST on PR #12"
    )
    assert "gh-with-backoff.sh" in run, (
        "the retry helper lives in scripts/gh-with-backoff.sh; sourcing it "
        "beats duplicating it verbatim in both steps"
    )
    assert "::notice::" in run, (
        "the fallback PR comment succeeding while inline posts failed is "
        "worth a notice, not a warning: findings ARE visible"
    )
    # Zero inline posts means the findings are invisible; the fallback is
    # the only remaining visibility path and its failure must fail the
    # step, not exit 0.
    assert "::error::" in run, (
        "the post step needs a loud error path when findings cannot be "
        "posted anywhere (inline posts failed AND the fallback failed)"
    )
    assert "exit 1" in run, (
        "when the fallback PR comment also fails, the step must exit "
        "nonzero: that is the invisible-findings failure mode from "
        "issue #15"
    )


def test_backoff_script_retries_only_transient_failures() -> None:
    script = REPO_ROOT / "scripts" / "gh-with-backoff.sh"
    assert script.exists(), (
        "scripts/gh-with-backoff.sh must exist: both workflow steps source "
        "it for their retry policy"
    )
    text = script.read_text(encoding="utf-8")
    assert "transient_gh_failure" in text, (
        "retries must be gated on a transient-failure check so permanent "
        "client errors (404/422) fail fast instead of burning sleeps"
    )
    assert re.search(r"rate limit", text, re.IGNORECASE), (
        "the transient check must recognise the secondary rate limit that "
        "caused issue #15"
    )
    assert "GH_RETRY_MAX_ATTEMPTS" in text, (
        "the retry budget must be a named, overridable knob"
    )
    assert "duplicate" in text, (
        "the script must state the accepted risk that a retried create "
        "can duplicate its output (visible duplicate beats invisible "
        "findings)"
    )
