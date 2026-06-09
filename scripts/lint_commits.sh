#!/usr/bin/env bash
# lint_commits.sh — Conventional Commits gate (CI-R1 item 15, QUAL-R12(b)).
#
# Invoked by `just lint-commits`. Lints every commit message in the PR range
# against the Conventional Commits v1.0 spec, scoped to the repo's documented
# prefixes (see CONTRIBUTING.md §4). A non-conforming message FAILS the build —
# the semver tag + CHANGELOG.md are derived from the conforming log by the
# release pipeline, so a malformed subject would corrupt the changelog.
#
# Validated shape (the subject line / first line of each commit):
#
#     type(scope)!: description
#     └┬─┘└─┬──┘│ └────┬─────┘
#      │    │   │      └ a non-empty, lower-case-initial, no-trailing-period summary
#      │    │   └ optional `!` marking a breaking change
#      │    └ optional `(scope)` — non-empty, no whitespace/parens/newlines inside
#      └ a type from the allow-list below
#
#   - A merge commit ("Merge ...") is exempt (CI may include them in the range).
#   - A `BREAKING CHANGE:`/`BREAKING-CHANGE:` footer is also a valid breaking marker
#     (Conventional Commits v1.0), independent of the `!`.
#
# Range selection (first that resolves wins):
#   1. $COMMIT_RANGE                       — explicit override (e.g. "A..B" or "A..").
#   2. PR base..HEAD                       — from CI's base ref
#                                            (GITHUB_BASE_REF / GITEA_BASE_REF /
#                                            FORGEJO base, github.event PR base).
#   3. origin/<default>..HEAD              — local fallback vs the upstream default branch.
#   4. HEAD                                — last resort: lint just the tip commit.
#
# Strict-mode, POSIX-friendly bash. stdout carries the human report; a non-zero
# exit is the gate signal. SEC-R13 spirit: if the gate cannot determine WHAT to
# lint it still lints SOMETHING (HEAD) rather than passing silently.

set -euo pipefail

# --- allow-list of types (CONTRIBUTING.md §4 / QUAL-R12) ------------------------------------------
# feat | fix | docs | style | refactor | perf | test | chore | ci | build
readonly ALLOWED_TYPES='feat|fix|docs|style|refactor|perf|test|chore|ci|build'

# --- conventional-commit subject grammar ---------------------------------------------------------
# type, optional (scope) with no whitespace/parens inside, optional `!`, `: `, then a non-empty
# description. ERE; the leading `^` anchors the whole subject line.
readonly CC_SUBJECT_RE="^(${ALLOWED_TYPES})(\([^()[:space:]]+\))?(!)?: .+"

log()  { printf '%s\n' "$*"; }
err()  { printf 'lint-commits: %s\n' "$*" >&2; }

# Resolve the inclusive commit range to lint, echoed as a `git log` range argument.
resolve_range() {
  # 1) explicit override.
  if [ -n "${COMMIT_RANGE:-}" ]; then
    printf '%s' "${COMMIT_RANGE}"
    return 0
  fi

  # 2) CI-provided PR base ref (GitHub Actions / Forgejo / Gitea all set one of these).
  local base_ref=""
  for v in "${GITHUB_BASE_REF:-}" "${GITEA_BASE_REF:-}" "${FORGEJO_BASE_REF:-}"; do
    if [ -n "$v" ]; then base_ref="$v"; break; fi
  done
  if [ -n "$base_ref" ]; then
    # Prefer the fetched remote-tracking ref; fall back to a local ref of the same name.
    local resolved=""
    for cand in "origin/${base_ref}" "${base_ref}"; do
      if git rev-parse --verify --quiet "${cand}^{commit}" >/dev/null 2>&1; then
        resolved="$cand"; break
      fi
    done
    if [ -n "$resolved" ]; then
      # Use the merge-base so we lint only commits unique to this PR head.
      local mb
      if mb="$(git merge-base "$resolved" HEAD 2>/dev/null)" && [ -n "$mb" ]; then
        printf '%s..HEAD' "$mb"
        return 0
      fi
      printf '%s..HEAD' "$resolved"
      return 0
    fi
  fi

  # 3) local fallback: upstream default branch's merge-base, if discoverable.
  local default_branch=""
  if default_branch="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)"; then
    : # e.g. "origin/main"
  elif git rev-parse --verify --quiet origin/main^{commit} >/dev/null 2>&1; then
    default_branch="origin/main"
  fi
  if [ -n "$default_branch" ]; then
    local mb
    if mb="$(git merge-base "$default_branch" HEAD 2>/dev/null)" && [ -n "$mb" ] \
        && [ "$mb" != "$(git rev-parse HEAD 2>/dev/null)" ]; then
      printf '%s..HEAD' "$mb"
      return 0
    fi
  fi

  # 4) last resort: just the tip commit (never pass silently for lack of a range).
  printf 'HEAD'
}

# Validate a single commit's subject line. Returns 0 if OK, 1 with a reason on stderr otherwise.
# $1 = short sha (for reporting), $2 = full subject line.
lint_subject() {
  local sha="$1" subject="$2"

  # Merge commits are not authored conventional commits; exempt them.
  case "$subject" in
    "Merge "*) return 0 ;;
  esac

  # Empty subject is never valid.
  if [ -z "$subject" ]; then
    err "${sha}: empty commit subject"
    return 1
  fi

  if ! printf '%s' "$subject" | grep -Eq "$CC_SUBJECT_RE"; then
    err "${sha}: not a Conventional Commit: \"${subject}\""
    err "        expected: <type>(<scope>)!: <description>"
    err "        type ∈ {${ALLOWED_TYPES//|/, }}; scope optional; '!' marks a breaking change"
    return 1
  fi

  # Reject a trailing period on the description (Conventional Commits style guidance).
  if printf '%s' "$subject" | grep -Eq '\.$'; then
    err "${sha}: description must not end with a period: \"${subject}\""
    return 1
  fi

  # Reject an empty description after the colon (defensive; the RE requires one char,
  # but a single space slips through `: .+` only with content — keep the check explicit).
  local desc="${subject#*: }"
  if [ -z "${desc//[[:space:]]/}" ]; then
    err "${sha}: empty description after type/scope: \"${subject}\""
    return 1
  fi

  return 0
}

main() {
  if ! command -v git >/dev/null 2>&1; then
    err "git is not installed — the gate cannot run and MUST NOT pass silently."
    exit 1
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    err "not inside a git work tree — cannot determine the commit range."
    exit 1
  fi

  local range
  range="$(resolve_range)"
  log "lint-commits: validating range '${range}' (Conventional Commits v1.0, QUAL-R12)"

  # Collect "<short-sha> <subject>" per commit, newline-separated, oldest→newest.
  # %x09 (tab) cleanly separates the sha from a subject that may contain spaces.
  local lines
  if ! lines="$(git log --no-merges --reverse --format='%h%x09%s' "$range" 2>/dev/null)"; then
    # An unresolvable range (e.g. shallow clone missing the base) — fail loudly.
    err "could not read commits for range '${range}' (is the base fetched / clone deep enough?)"
    exit 1
  fi

  if [ -z "$lines" ]; then
    log "lint-commits: no non-merge commits in range — nothing to lint."
    exit 0
  fi

  local total=0 bad=0
  while IFS=$'\t' read -r sha subject; do
    [ -z "$sha" ] && continue
    total=$((total + 1))
    if lint_subject "$sha" "$subject"; then
      log "  ok   ${sha}  ${subject}"
    else
      log "  FAIL ${sha}  ${subject}"
      bad=$((bad + 1))
    fi
  done <<< "$lines"

  if [ "$bad" -gt 0 ]; then
    err "${bad}/${total} commit message(s) violate Conventional Commits — see CONTRIBUTING.md §4."
    exit 1
  fi

  log "lint-commits: all ${total} commit message(s) conform. ✓"
}

main "$@"
