#!/usr/bin/env bash
# Shared gh api retry policy for the code-review workflow steps.
# Source this file; do not execute it. Both helpers take the full
# command (gh api ...) as their arguments and run it as given.
GH_RETRY_MAX_ATTEMPTS="${GH_RETRY_MAX_ATTEMPTS:-6}"
# A non-numeric budget makes the integer comparison below error out, which
# reads as "not exceeded" and leaves the loop bounded only by the job
# timeout (#22), so fall back to the default instead.
[[ "$GH_RETRY_MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || GH_RETRY_MAX_ATTEMPTS=6

# Retry only failures that look transient (rate limits, 5xx, network
# errors); permanent client errors fail fast with their stderr instead
# of burning sleeps. "429"/"too many requests"/"timeout" cover the bodies
# gh prints for the same secondary-limit and i/o classes (#22).
transient_gh_failure() {
  grep -qiE 'rate limit|retry after|too many requests|429|HTTP 5[0-9][0-9]|connection reset|timed? out|timeout|EOF|dial tcp|no such host|deadline exceeded'
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
# holds stderr for the retry decision without a second API call. The temp
# file is removed explicitly on every exit path instead of via a RETURN
# trap, so no trap state outlives the call in the sourcing shell (#22).
gh_read_with_backoff() {
  local attempt=1 out err_file
  err_file=$(mktemp) || return 1
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
