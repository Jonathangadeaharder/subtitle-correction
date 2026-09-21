"""Pin the hardened config pin and coverage artifact handling.

Issue #24 (filed from the VocabLevels PR #181 review, where the same
two defects were fixed) pins two failure modes in the privileged
`sonarqube` job of .github/workflows/sonarqube.yml:

- the config pin step used `rm -f`, which cannot remove a directory:
  a PR committing `sonar-project.properties/` aborted the pin step
  with a confusing `rm` diagnostic instead of the intended config
  fallback error;
- `gh run download` bulk-extracted the whole PR-controlled artifact
  into the token-bearing workspace before any check could run: no
  size bound, every archive member landed in the checkout, and the
  coverage report the privileged scanner parses was never validated.
"""

import re

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "sonarqube.yml"

PIN_STEP = "Pin scanner config to main"
DOWNLOAD_STEP = "Download coverage report from the CI run"


def step_run_text(name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["sonarqube"]["steps"]
    matching = [s for s in steps if s.get("name") == name]
    assert matching, f"step {name!r} not found in sonarqube.yml"
    return matching[0]["run"]


def test_pin_step_removes_directory_and_symlink_shapes() -> None:
    run = step_run_text(PIN_STEP)
    assert "rm -rf sonar-project.properties" in run, (
        "the pin step must use `rm -rf`: `rm -f` cannot remove a "
        "directory, so a PR committing sonar-project.properties/ aborts "
        "the step with an rm diagnostic instead of the intended "
        "config-fallback error (issue #24)"
    )
    assert not re.search(r"rm -f sonar-project\.properties", run), (
        "no `rm -f sonar-project.properties` may remain: it fails on a "
        "directory at that path (issue #24)"
    )


def test_coverage_step_never_bulk_extracts_the_artifact() -> None:
    run = step_run_text(DOWNLOAD_STEP)
    assert "gh run download" not in run, (
        "`gh run download` extracts every archive member inside the "
        "privileged workspace before any check can run; the artifact is "
        "PR-controlled input to a token-bearing job (issue #24)"
    )
    assert "actions/artifacts/" in run and "/zip" in run, (
        "the artifact must be fetched as a raw zip via the artifacts API, "
        "not bulk-extracted by gh (issue #24)"
    )
    assert "size_in_bytes" in run, (
        "the artifact size must be gated on its reported size_in_bytes "
        "before the download (issue #24)"
    )
    assert "256 * 1024 * 1024" in run, (
        "both the artifact gate and the member extraction must be bounded to 256 MiB (issue #24)"
    )
    assert '--jq \'.artifacts[] | select(.name == "coverage")' in run, (
        "the artifact lookup must select the coverage artifact by name "
        "and must paginate the artifact list (issue #24)"
    )


def test_coverage_step_extracts_only_the_report_member_bounded() -> None:
    run = step_run_text(DOWNLOAD_STEP)
    assert '"coverage.xml" not in archive.namelist()' in run, (
        "the zip must be checked for the coverage.xml member before any "
        "extraction, so a memberless artifact fails loudly (issue #24)"
    )
    assert 'archive.open("coverage.xml")' in run, (
        "only the coverage.xml member may be extracted; sibling entries "
        "must never land in the workspace (issue #24)"
    )
    assert re.search(r"src\.read\(1024 \* 1024\)", run), (
        "the member extraction must be a bounded chunked read, not a "
        "single unbounded extract (a small zip can carry a multi-GB "
        "member; issue #24)"
    )
    assert 'open("coverage.xml", "wb")' in run, (
        "the report must be written to a fixed workspace-root path, "
        "outside sonar.sources=subtitle_correction, never to a path "
        "taken from the archive (issue #24)"
    )


def test_coverage_step_validates_the_report_before_scanning() -> None:
    run = step_run_text(DOWNLOAD_STEP)
    assert 'b"\\x00" in data' in run, (
        "NUL bytes must be rejected before parsing: a BOM-less UTF-16/32 "
        "report is valid UTF-8 to expat, which auto-detects the real "
        "encoding and expands a hidden DTD (issue #24)"
    )
    assert '.decode("utf-8")' in run, (
        "the report must be forced through a UTF-8 decode before the scanner parses it (issue #24)"
    )
    assert "<!DOCTYPE" in run and "<!ENTITY" in run, (
        "DTD and entity declarations must be rejected: the report is "
        "parsed by a privileged process (issue #24)"
    )
    assert "ET.fromstring" in run, (
        "the report must actually be parsed, not just pattern-matched: "
        "malformed XML must fail the step here, not inside the scanner "
        "(issue #24)"
    )
    assert "os.path.isabs" in run and '".." in' in run, (
        "class filenames must be validated relative and contained in the "
        "workspace before the scanner resolves them (issue #24)"
    )
    assert 'iter("class")' in run, (
        "a class-less report is an empty report: it must fail loudly "
        "instead of scanning nothing (issue #24)"
    )
    assert "python3 -I" in run, (
        "the extraction and validation must run trusted stdlib python in "
        "isolated mode, never repository code from the untrusted tree "
        "(ADR 0001 invariant: the job never executes the tree)"
    )


def test_coverage_step_cleans_up_the_downloaded_zip() -> None:
    run = step_run_text(DOWNLOAD_STEP)
    assert "ZIP_PATH: ${{ runner.temp }}" in WORKFLOW_PATH.read_text(encoding="utf-8"), (
        "the raw zip must be staged outside the workspace in runner.temp, "
        "so no PR-controlled archive ever lands inside the checkout "
        "(issue #24)"
    )
    assert "trap 'rm -f \"$ZIP_PATH\"' EXIT" in run, (
        "the staged zip must be removed even when validation fails, so a "
        "cancelled retry does not inherit stale artifact bytes (issue #24)"
    )
