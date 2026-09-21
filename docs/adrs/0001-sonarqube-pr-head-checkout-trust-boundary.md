# SonarQube PR-head checkout trust boundary

* Status: accepted
* Date: 2026-09-21
* Issue: #14 (follow-up from the review of PR #12, comment 4058466263)

## Context and Problem Statement

PR #12 moved the SonarQube scan behind `workflow_run` so that fork and
dependabot PRs can be analyzed with secrets available. The cost: the
privileged `sonarqube` job (SONAR_TOKEN and a checks:write GITHUB_TOKEN in
scope) checks out the untrusted PR head at
`github.event.workflow_run.head_sha`. The scanner reads that tree; nothing
in the job executes it today, but any future step added to the job that
touches the tree (a lint, a build, a test) would run PR-supplied code with
secrets in scope.

## Decision Drivers

* `SONAR_TOKEN` must never be reachable by code a PR author controls.
* Fork and dependabot PRs must still get automatic analysis (the reason PR
  #12 exists).
* The invariant "this job never executes the tree" must survive future
  edits to the workflow, not just hold today.

## Considered Options

1. **The two-workflow split from
   jenkinsci/jira-plugin's
   docs/adr/0001-sonarcloud-analysis-for-fork-prs.md** — an unprivileged
   `pull_request` workflow builds/tests and uploads artifacts; a trusted
   `workflow_run` job downloads the artifacts, checks out the head
   read-only, and submits the analysis via the scanner CLI. This is the
   pattern this repository already implements in substance: the
   unprivileged CI workflow (`ci.yml`, `permissions: contents: read`) runs
   the tests and uploads `coverage.xml`; the privileged job never runs
   tests and scans via `SonarSource/sonarqube-scan-action` (the scanner
   CLI, not a build tool). Crucially, the reference ADR's trusted job also
   checks out the fork head: "Downloads the artifacts, checks out the
   fork's head commit read-only, and submits analysis". Adopting the
   pattern cannot remove the checkout, because a scanner needs the
   sources.
2. **Upload a source archive from the unprivileged CI job** instead of
   checking out the PR head in the trusted job. Rejected: extracting an
   attacker-crafted archive with secrets in scope is a worse parsing
   surface than the audited, digest-pinned `actions/checkout` (tar path
   traversal vs. a battle-tested action), and the residual risk is
   unchanged either way: the trusted job still reads PR-authored files.
3. **`pull_request_target`** — rejected: it is the textbook pwn-request
   setup and PR #12 exists precisely to leave it behind.
4. **Pin the residual risk in an ADR** with an enforceable structural
   tripwire.

## Decision Outcome

Chosen option: **4, plus option 1's already-present substance**. The
checkout stays, and the residual risk is accepted explicitly under the
following invariants, which `tests/test_sonar_workflow_trust_boundary.py`
enforces structurally:

* The privileged `sonarqube` job's step list is pinned to an allowlist. Any
  new step in that job fails the test until the ADR is re-reviewed and the
  allowlist is consciously updated. That is the tripwire for "a future
  step touches the tree".
* The job never executes repository code: every step only reads the tree
  (scanner CLI, `git show` of the pinned config) or talks to the API.
* The checkout is pinned by commit digest with `persist-credentials: false`
  so the GITHUB_TOKEN never lands in the untrusted tree's `.git/config`.
* The scanner config (`sonar-project.properties`) is taken from
  `origin/main`, never from the PR.
* Attacker-influenced strings (branch names, PR numbers, base refs) are
  validated against strict regexes before any use in `gh api` calls.
* The scanner action itself is pinned by commit digest.

## Consequences

* Fork and dependabot PRs keep automatic analysis with secrets in scope of
  the trusted context only.
* Residual accepted risk, shared with the reference pattern: the trusted
  job reads PR-authored source and a PR-influenced `coverage.xml` produced
  by the unprivileged CI job. It does not execute them; a crafted file
  exploiting a bug in `sonar-scanner` itself is the theoretical risk every
  project using this pattern carries.
* Adding a step to the privileged job is a security review event: update
  this ADR and the test allowlist together, in the same PR.
