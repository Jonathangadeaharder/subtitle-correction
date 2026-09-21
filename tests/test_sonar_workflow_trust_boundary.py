"""Pin the trust boundary of the privileged SonarQube scan job.

The `sonarqube` job in .github/workflows/sonarqube.yml checks out the
untrusted PR head while secrets are in scope. The residual risk and the
invariants that keep it acceptable are pinned in
docs/adrs/0001-sonarqube-pr-head-checkout-trust-boundary.md. These tests
make the structural half of that ADR enforceable: any new step added to
the privileged job fails the allowlist and forces a re-read of the ADR.
"""

import re

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "sonarqube.yml"
ADR_PATH = (
    REPO_ROOT / "docs" / "adrs" / "0001-sonarqube-pr-head-checkout-trust-boundary.md"
)

PRIVILEGED_JOB = "sonarqube"
PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")

# Steps allowed in the privileged job, in order. The job holds secrets
# (SONAR_TOKEN, checks:write) while the untrusted PR tree is on disk, so
# every step here must only read the tree, never execute it. Adding a step
# requires re-reviewing the ADR first.
ALLOWED_STEPS = [
    "Derive scan arguments",
    "actions/checkout",
    "Pin scanner config to main",
    "Download coverage report from the CI run",
    "SonarQube Scan",
    "Print failing quality gate conditions",
    "Report result to the PR head",
]


def load_privileged_job() -> dict:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["jobs"][PRIVILEGED_JOB]


def test_adr_pins_the_residual_risk_of_the_pr_head_checkout() -> None:
    assert ADR_PATH.exists(), (
        "docs/adrs/0001-sonarqube-pr-head-checkout-trust-boundary.md must "
        "exist: the privileged job checks out the untrusted PR head, and the "
        "residual risk plus its invariants live in that ADR, not in the "
        "workflow file alone"
    )
    adr = ADR_PATH.read_text(encoding="utf-8")
    assert "Status" in adr and "accepted" in adr
    # The ADR must name the invariant the allowlist test enforces.
    assert "never execute" in adr.lower()
    # The ADR must name the reference pattern it was compared against.
    assert "jenkinsci/jira-plugin" in adr


def test_privileged_job_steps_are_pinned_to_the_adr_allowlist() -> None:
    job = load_privileged_job()
    step_labels = []
    for step in job["steps"]:
        if step.get("name"):
            step_labels.append(step["name"])
        else:
            step_labels.append(step["uses"].split("@")[0])
    assert step_labels == ALLOWED_STEPS, (
        f"Step list of the privileged '{PRIVILEGED_JOB}' job drifted from the "
        "allowlist pinned in docs/adrs/0001. A new step in a job that holds "
        "secrets while the untrusted PR tree is checked out is a security "
        "review event: update the ADR first, then this allowlist."
    )


def test_privileged_checkout_is_pinned_and_persists_no_credentials() -> None:
    job = load_privileged_job()
    checkouts = [
        step
        for step in job["steps"]
        if "uses" in step and step["uses"].startswith("actions/checkout")
    ]
    assert len(checkouts) == 1, "expected exactly one checkout in the scan job"
    assert PINNED_ACTION.match(checkouts[0]["uses"]), (
        "the untrusted-tree checkout must be pinned by commit digest, not a "
        "mutable tag"
    )
    assert checkouts[0]["with"]["persist-credentials"] is False, (
        "the GITHUB_TOKEN must never be persisted into the untrusted tree's "
        ".git/config"
    )


def test_privileged_scanner_action_is_pinned_by_digest() -> None:
    job = load_privileged_job()
    scan_steps = [
        step
        for step in job["steps"]
        if "uses" in step and "sonarqube-scan-action" in step["uses"]
    ]
    assert len(scan_steps) == 1
    assert PINNED_ACTION.match(scan_steps[0]["uses"]), (
        "the scanner that reads the untrusted tree must be pinned by commit "
        "digest"
    )
