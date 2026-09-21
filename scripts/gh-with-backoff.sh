#!/usr/bin/env bash
# Shared gh api retry policy for the code-review workflow steps.
# Source this file; do not execute it. Both helpers take the full
# command (gh api ...) as their arguments and run it as given.
GH_RETRY_MAX_ATTEMPTS="${GH_RETRY_MAX_ATTEMPTS:-6}"

# Retry only failures that look transient (rate limits, 5xx); permanent
# client errors fail fast with their stderr instead of burning sleeps.
transient_gh_failure() {
  grep -qiE 'rate limit|retry after|HTTP 5[0-9][0-9]'
}

# A retried create can duplicate its output when the first attempt
# reached GitHub but the response was lost. Accepted: a visible
# duplicate beats the invisible-findings failure of issue #15.
gh_with_backoff() {
  local attempt=1 err
  while :; do
    if err=$("$@" 2>&1 >/dev/null); then
      return 0
    fi
    if ! printf '%s' "$err" | transient_gh_failure ||
      [ "$attempt" -ge "$GH_RETRY_MAX_ATTEMPTS" ]; then
      printf '%s\n' "$err" >&2
      return 1
    fi
    sleep $((2 ** attempt))
    attempt=$((attempt + 1))
  done
}

# Read variant: prints stdout for the caller to capture; a temp file
# holds stderr for the retry decision without a second API call.
gh_read_with_backoff() {
  local attempt=1 out err_file
  err_file=$(mktemp)
  while :; do
    if out=$("$@" 2>"$err_file"); then
      rm -f "$err_file"
      printf '%s' "$out"
      return 0
    fi
    if ! transient_gh_failure < "$err_file" ||
      [ "$attempt" -ge "$GH_RETRY_MAX_ATTEMPTS" ]; then
      cat "$err_file" >&2
      rm -f "$err_file"
      return 1
    fi
    sleep $((2 ** attempt))
    attempt=$((attempt + 1))
  done
}
