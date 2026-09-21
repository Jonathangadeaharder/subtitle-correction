"""Pin the CI/Sonar concurrency refinements from issue #16.

Three advisory findings from the review of PR #12, accepted as one polish
batch:

- CI's cancel-in-progress also cancels push runs to main; a cancelled
  main run still fires the SonarQube workflow_run with conclusion
  'cancelled', which fails the scan by design, so a rapid-push window
  leaves the earlier commit unanalyzed.
- SonarQube's concurrency group is keyed only by repository and head
  SHA, so a PR-event and push-event scan of the same SHA can cancel each
  other.
- The zero-rows error advises re-running the workflow, but the coverage
  artifact expires after one day, so a later re-run fails at the
  download; the guidance must prefer a fresh push.
"""

import re

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SONAR_PATH = REPO_ROOT / ".github" / "workflows" / "sonarqube.yml"


def test_ci_cancellation_is_limited_to_pull_requests() -> None:
    ci = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    cancel = ci["concurrency"]["cancel-in-progress"]
    if isinstance(cancel, bool):
        raise AssertionError(
            "cancel-in-progress must be the expression "
            "${{ github.event_name == 'pull_request' }}, not a bare "
            f"boolean (got {cancel!r}): a bare true cancels rapid pushes "
            "to main, and the cancelled run makes SonarQube fail by design"
        )
    assert "github.event_name == 'pull_request'" in cancel


def test_sonar_concurrency_group_separates_event_kinds() -> None:
    sonar = yaml.safe_load(SONAR_PATH.read_text(encoding="utf-8"))
    group = sonar["concurrency"]["group"]
    assert "github.event.workflow_run.event" in group, (
        "the concurrency group must key on the triggering event kind: "
        "without it a PR-event and push-event scan of the same head SHA "
        f"share a group and can cancel each other (got {group!r})"
    )


def test_re_run_guidance_names_the_artifact_expiry() -> None:
    text = SONAR_PATH.read_text(encoding="utf-8")
    zero_rows_hint = re.search(
        r"No pull requests found for scanned SHA[^\"]*", text
    )
    assert zero_rows_hint, "the zero-rows error message must still exist"
    # The artifact (coverage, retention-days: 1) expires, so a later
    # re-run cannot succeed: the guidance must prefer a fresh push.
    assert "expires" in text, (
        "the re-run guidance must say the coverage artifact expires after "
        "one day, so a re-run later than that fails at the download"
    )
