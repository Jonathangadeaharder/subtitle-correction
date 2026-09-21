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
import subprocess
import tempfile

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
    assert "GITHUB_STEP_SUMMARY" in run, (
        "a failed fetch must be recorded somewhere durable (step summary), "
        "not just a warning line in a green step log: stale comments "
        "accumulate invisibly across repeated failures"
    )
    assert "top-level" in run, (
        "the in_reply_to_id filter narrows resolution to top-level "
        "comments; a comment in the YAML must say so, or a future edit "
        "reads it as an oversight"
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
    assert re.search(r"connection reset", text, re.IGNORECASE), (
        "the transient check must also survive network-level failures "
        "(connection reset, timeout, DNS): gh reports those differently "
        "from HTTP rate limits"
    )
    assert "trap" in text, (
        "the read helper must clean up its temp file even when the retry "
        "loop is interrupted"
    )
    assert "GH_RETRY_MAX_ATTEMPTS" in text, (
        "the retry budget must be a named, overridable knob"
    )
    assert "duplicate" in text, (
        "the script must state the accepted risk that a retried create "
        "can duplicate its output (visible duplicate beats invisible "
        "findings)"
    )


def _fake_gh(tmp: str, behavior: str) -> None:
    # A fake gh that counts invocations and fails per `behavior`:
    # "transient_once" fails with a rate-limit message on call 1 then
    # succeeds; "permanent" always fails with a 422.
    calls_file = f"{tmp}/calls"
    gh = Path(tmp) / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        f'CALLS_FILE="{calls_file}"\n'
        'n=$(cat "$CALLS_FILE" 2>/dev/null || echo 0)\n'
        'n=$((n + 1)); echo "$n" > "$CALLS_FILE"\n'
        # Argue like the real gh api: one endpoint positional, then flags.
        'shift 2\n'
        'for a in "$@"; do\n'
        '  case "$a" in -*) ;; *)\n'
        '    echo "accepts 1 arg(s), received more" >&2; exit 1;;\n'
        "  esac\n"
        "done\n"
        f'case "{behavior}" in\n'
        "  transient_once)\n"
        '    [ "$n" -gt 1 ] && echo \'{"ok": true}\' && exit 0\n'
        '    echo "gh: HTTP 403: secondary rate limit exceeded, retry after 30s" >&2\n'
        "    exit 1;;\n"
        "  always_transient)\n"
        '    echo "gh: HTTP 403: secondary rate limit exceeded, retry after 30s" >&2\n'
        "    exit 1;;\n"
        "  permanent)\n"
        '    echo "gh: HTTP 422: Validation Failed" >&2\n'
        "    exit 1;;\n"
        "  too_many_requests_once)\n"
        '    [ "$n" -gt 1 ] && echo \'{"ok": true}\' && exit 0\n'
        '    echo "gh: HTTP 429: Too Many Requests" >&2\n'
        "    exit 1;;\n"
        "  timeout_headers_once)\n"
        '    [ "$n" -gt 1 ] && echo \'{"ok": true}\' && exit 0\n'
        '    echo "Get https://api.github.com/repos/o/r: net/http: '
        'timeout awaiting response headers" >&2\n'
        "    exit 1;;\n"
        "esac\n"
    )
    gh.chmod(0o755)


def _run_helper(tmp: str, call: str) -> subprocess.CompletedProcess:
    script = (
        f'export PATH="{tmp}:$PATH"\n'
        f'. "{REPO_ROOT / "scripts" / "gh-with-backoff.sh"}"\n'
        f"{call}\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_backoff_helper_retries_transient_failures() -> None:
    # The helper receives the full command (gh api ...) and must execute
    # it as-is. Run 35555738571 failed because the helper re-prefixed
    # "gh api" onto an argument list that already contained it.
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "transient_once")
        result = _run_helper(
            tmp, "gh_with_backoff gh api repos/o/r/pulls/1/comments || exit 1"
        )
        calls = (Path(tmp) / "calls").read_text().strip()
        assert result.returncode == 0, result.stderr
        assert calls == "2", (
            "a transient rate-limit failure must be retried, not failed fast"
        )


def test_backoff_helper_fails_fast_on_permanent_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "permanent")
        result = _run_helper(
            tmp, "gh_with_backoff gh api repos/o/r/pulls/1/comments"
        )
        calls = (Path(tmp) / "calls").read_text().strip()
        assert result.returncode != 0
        assert calls == "1", (
            "a permanent client error (422) must not burn retry sleeps"
        )
        assert "HTTP 422" in result.stderr, (
            "the helper must surface the underlying gh stderr, not swallow it"
        )


def test_backoff_read_helper_returns_stdout_after_retry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "transient_once")
        result = _run_helper(
            tmp,
            "COMMENTS=$(gh_read_with_backoff gh api repos/o/r/pulls/1/comments)"
            ' || exit 1; echo "DATA:$COMMENTS"\nexit 0',
        )
        calls = (Path(tmp) / "calls").read_text().strip()
        assert result.returncode == 0, result.stderr
        assert calls == "2"
        assert "DATA:{\"ok\": true}" in result.stdout, (
            "the read helper must print the payload to stdout for the "
            "caller's command substitution to capture"
        )


def test_backoff_helper_gives_up_after_max_attempts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "always_transient")
        result = _run_helper(
            tmp,
            "export GH_RETRY_MAX_ATTEMPTS=2\n"
            "gh_with_backoff gh api repos/o/r/pulls/1/comments",
        )
        calls = (Path(tmp) / "calls").read_text().strip()
        assert result.returncode != 0
        assert calls == "2", (
            "an always-transient failure must stop at GH_RETRY_MAX_ATTEMPTS"
        )
        assert "HTTP 403" in result.stderr, (
            "giving up must still surface the last gh stderr"
        )


def test_every_backoff_call_site_passes_the_full_command() -> None:
    # Run 35556097175: one call site passed the bare endpoint, the helper
    # executed it as a command (exit 127), and the resolve step skipped
    # resolving every previous round's comments.
    for name in (RESOLVE_STEP, POST_STEP):
        run = step_run_text(name)
        first_words = re.findall(r"gh_(?:read_)?with_backoff\s+(\S+)", run)
        assert first_words, f"{name} must use the retry helpers"
        assert first_words == ["gh"] * len(first_words), (
            f"{name}: retry-helper call sites must pass the full command "
            f"('gh api <endpoint> ...'), got {first_words}"
        )


def test_backoff_retries_a_bare_429_body() -> None:
    # Advisory from PR #21: the matcher knew "rate limit" and "retry after"
    # but a bare "HTTP 429: Too Many Requests" body failed fast instead of
    # retrying, which is the same secondary-limit class as issue #15.
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "too_many_requests_once")
        result = _run_helper(
            tmp, "gh_with_backoff gh api repos/o/r/pulls/1/comments || exit 1"
        )
        calls = (Path(tmp) / "calls").read_text().strip()
        assert result.returncode == 0, result.stderr
        assert calls == "2", "a 429 body must be retried"


def test_backoff_retries_net_http_timeout_headers() -> None:
    # Advisory from PR #21: "timed? out" matched "time out"/"timed out"
    # but not Go's "net/http: timeout awaiting response headers".
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "timeout_headers_once")
        result = _run_helper(
            tmp,
            "COMMENTS=$(gh_read_with_backoff gh api repos/o/r/pulls/1/comments)"
            " || exit 1; echo \"DATA:$COMMENTS\"\nexit 0",
        )
        calls = (Path(tmp) / "calls").read_text().strip()
        assert result.returncode == 0, result.stderr
        assert calls == "2", "an i/o-level timeout must be retried"


def test_read_helper_does_not_leak_its_return_trap() -> None:
    # Advisory from PR #21: the RETURN trap set inside gh_read_with_backoff
    # persisted in the sourcing shell, firing a no-op `rm -f ""` on every
    # later function return in that shell.
    with tempfile.TemporaryDirectory() as tmp:
        _fake_gh(tmp, "transient_once")
        result = _run_helper(
            tmp,
            "COMMENTS=$(gh_read_with_backoff gh api repos/o/r/pulls/1/comments)"
            ' || exit 1\n'
            'remaining=$(trap -p RETURN)\n'
            'echo "TRAP:[$remaining]"\n'
            "exit 0",
        )
        assert result.returncode == 0, result.stderr
        assert "TRAP:[]" in result.stdout, (
            "the read helper must restore the previous RETURN trap"
        )


def test_non_numeric_retry_budget_is_normalised() -> None:
    # Advisory from PR #21: a non-numeric GH_RETRY_MAX_ATTEMPTS makes the
    # integer comparison error out, so the loop is bounded only by the job
    # timeout. The default must be restored instead.
    script = (
        "export GH_RETRY_MAX_ATTEMPTS=abc\n"
        f'. "{REPO_ROOT / "scripts" / "gh-with-backoff.sh"}"\n'
        'echo "BUDGET:$GH_RETRY_MAX_ATTEMPTS"\n'
        "exit 0\n"
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert "BUDGET:6" in result.stdout, (
        f"a non-numeric budget must fall back to 6, got {result.stdout!r}"
    )
