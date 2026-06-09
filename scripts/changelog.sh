#!/usr/bin/env bash
# changelog.sh — generate/update CHANGELOG.md from the Conventional-Commits log.
#
# Invoked by the Justfile `changelog` recipe. Implements CI-R10 step 3 + QUAL-R12(c): every release
# refreshes CHANGELOG.md from the conforming commit log (Conventional Commits), in Keep a Changelog
# format, deriving the section grouping (Added/Fixed/Changed/…) from each commit's `type`.
#
# Design choices (the tool is NOT baked into YAML — see the justfile `changelog` recipe):
#   * If `git-cliff` is on PATH it is used (richer config, semver-aware). It is the preferred engine.
#   * Otherwise a self-contained git-log fallback parses Conventional Commits directly so the gate
#     still runs with zero extra tooling. The fallback is dependency-free (git + coreutils only).
#   * IDEMPOTENT: re-running on an unchanged repo produces byte-identical output (no duplicated
#     entries, stable ordering). The whole file is regenerated from the commit log each run, so it
#     is a pure function of (git history, tags) — never appends.
#   * RELEASE_DRY_RUN=1 PRINTS the rendered changelog to stdout and writes NOTHING (CI-R10), so the
#     release dry-run can show the intended file without mutating the working tree or touching disk.
#
# Usage:  scripts/changelog.sh
#         RELEASE_DRY_RUN=1 scripts/changelog.sh        # print to stdout, write nothing
#         CHANGELOG_FILE=CHANGELOG.md scripts/changelog.sh

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

CHANGELOG_FILE="${CHANGELOG_FILE:-${WW_REPO_ROOT}/CHANGELOG.md}"
# Normalise to an absolute path so the dry-run/print logic and the write target agree regardless of
# whether a relative override was passed.
case "${CHANGELOG_FILE}" in
  /*) : ;;
  *)  CHANGELOG_FILE="${WW_REPO_ROOT}/${CHANGELOG_FILE}" ;;
esac

ww_have git || ww_die "git is required to derive the changelog from the commit log."
git -C "${WW_REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1 \
  || ww_die "not inside a git work tree (${WW_REPO_ROOT}) — cannot read the commit log."

# --- engine: git-cliff (preferred) ---------------------------------------------------------------
# git-cliff renders Keep-a-Changelog / Conventional-Commits output natively and is semver/tag aware.
# We let it write the file itself (or, under dry-run, emit to stdout) so its own config is honored.
render_with_git_cliff() {
  local out
  if ww_is_dry_run; then
    ww_log "DRY-RUN would regenerate ${CHANGELOG_FILE} via git-cliff (printing to stdout, no write)"
    ( cd "${WW_REPO_ROOT}" && git-cliff ) || return 1
    return 0
  fi
  ww_log "regenerating ${CHANGELOG_FILE} via git-cliff"
  # Render to a temp file first, then move into place atomically — a crashed git-cliff must not
  # leave a half-written CHANGELOG (keeps the step idempotent + crash-safe).
  out="$(mktemp "${TMPDIR:-/tmp}/ww-changelog.XXXXXX")"
  if ( cd "${WW_REPO_ROOT}" && git-cliff --output "${out}" ); then
    mv -f "${out}" "${CHANGELOG_FILE}"
    return 0
  fi
  rm -f "${out}"
  return 1
}

# --- engine: git-log fallback (dependency-free) --------------------------------------------------
# Parses Conventional Commits straight from `git log` and renders one Keep-a-Changelog block.
# Mapping of Conventional `type` -> Keep-a-Changelog section. Unknown/uncategorised types are
# dropped from the user-facing changelog (chore/ci/test/build/style/refactor/docs are internal —
# QUAL-R12(c) wants the release-relevant changes), EXCEPT `docs` which we surface under "Changed"-
# adjacent "Documentation". `feat!`/`fix!`/`type(scope)!` and a `BREAKING CHANGE:` trailer route to
# a leading "Changed (BREAKING)" group.
render_with_git_log() {
  # Initialise every accumulator to empty: under `set -u` a `+=` on an unset var would abort.
  local header=""
  local section_added="" section_changed="" section_fixed="" section_removed=""
  local section_security="" section_deprecated="" section_breaking=""
  local subject="" type="" breaking="" line="" scope=""

  # Collect commits since the most recent tag if one exists, else the whole history. A unit of
  # output is the [Unreleased] block — release.sh stamps the version when it cuts a tag; here we
  # always render the not-yet-released delta so the file is correct between releases too.
  local range=""
  local last_tag
  last_tag="$(git -C "${WW_REPO_ROOT}" describe --tags --abbrev=0 2>/dev/null || true)"
  [ -n "${last_tag}" ] && range="${last_tag}..HEAD"

  # %s = subject; %b = body (for BREAKING CHANGE trailer). Use an unambiguous record separator.
  local rec_sep=$'\x1e' fld_sep=$'\x1f'
  local raw
  raw="$(git -C "${WW_REPO_ROOT}" log --no-merges --pretty=format:"%s${fld_sep}%b${rec_sep}" ${range} 2>/dev/null || true)"

  while IFS= read -r -d "${rec_sep}" line; do
    # Each record may carry a leading newline left by the previous commit's multi-line body; strip
    # leading/trailing newlines so the subject regex anchors on the real `type(scope): …` text.
    line="${line#"${line%%[![:space:]]*}"}"
    [ -n "${line}" ] || continue
    subject="${line%%${fld_sep}*}"
    local body="${line#*${fld_sep}}"
    # Conventional Commit: type(scope)!: subject
    if [[ "${subject}" =~ ^([a-zA-Z]+)(\(([^\)]*)\))?(!)?:\ (.*)$ ]]; then
      type="${BASH_REMATCH[1],,}"
      scope="${BASH_REMATCH[3]}"
      breaking="${BASH_REMATCH[4]}"
      local desc="${BASH_REMATCH[5]}"
    else
      # Non-conforming subject — skip (the commit-lint gate keeps these out of main).
      continue
    fi

    local entry="- ${desc}"
    [ -n "${scope}" ] && entry="- **${scope}:** ${desc}"

    # A breaking change (either `!` or a BREAKING CHANGE body trailer) is always surfaced.
    if [ -n "${breaking}" ] || printf '%s' "${body}" | grep -qiE '(^|[[:space:]])BREAKING[ -]CHANGE'; then
      section_breaking+="${entry}"$'\n'
      continue
    fi

    case "${type}" in
      feat)            section_added+="${entry}"$'\n' ;;
      fix)             section_fixed+="${entry}"$'\n' ;;
      perf)            section_changed+="${entry}"$'\n' ;;
      revert)          section_removed+="${entry}"$'\n' ;;
      security|sec)    section_security+="${entry}"$'\n' ;;
      deprecate*)      section_deprecated+="${entry}"$'\n' ;;
      *)               : ;;  # chore/ci/test/build/style/refactor/docs — internal, omitted
    esac
  done <<< "${raw}"

  header="# Changelog

All notable changes to \`wattwise-core\` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses automated
semver derived from Conventional Commits.

## [Unreleased]
"

  # Emit only the non-empty sections, in Keep-a-Changelog canonical order, with stable sorting so
  # the output is a deterministic function of history (idempotent).
  local out="${header}"
  local s
  emit_section() {
    local title="$1" body="$2"
    [ -n "${body}" ] || return 0
    out+=$'\n'"### ${title}"$'\n'
    out+="$(printf '%s' "${body}" | sed '/^$/d' | sort)"$'\n'
  }
  emit_section "Changed (BREAKING)" "${section_breaking}"
  emit_section "Added"              "${section_added}"
  emit_section "Changed"            "${section_changed}"
  emit_section "Deprecated"         "${section_deprecated}"
  emit_section "Removed"            "${section_removed}"
  emit_section "Fixed"             "${section_fixed}"
  emit_section "Security"           "${section_security}"

  # If no conforming commits produced any section, keep a non-empty, valid file rather than an empty
  # one (a zero-content changelog would be a confusing release artifact).
  if [ "${out}" = "${header}" ]; then
    out+=$'\n'"### Added"$'\n'"- Initial development; no release-relevant changes recorded yet."$'\n'
  fi

  if ww_is_dry_run; then
    ww_log "DRY-RUN would write ${CHANGELOG_FILE} (printing to stdout, no write):"
    printf '%s' "${out}"
    return 0
  fi

  # Idempotent write: render to a temp file, then only replace the target if the bytes actually
  # differ (stable mtime + byte-identical output on no-op re-runs). Comparing files (not $(cat …))
  # avoids the trailing-newline stripping that command substitution would introduce.
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/ww-changelog.XXXXXX")"
  printf '%s' "${out}" > "${tmp}"
  if [ -f "${CHANGELOG_FILE}" ] && cmp -s "${tmp}" "${CHANGELOG_FILE}"; then
    rm -f "${tmp}"
    ww_log "${CHANGELOG_FILE} already up to date — no change."
    return 0
  fi
  mv -f "${tmp}" "${CHANGELOG_FILE}"
  ww_log "wrote ${CHANGELOG_FILE} (git-log fallback)."
}

# --- engine selection ----------------------------------------------------------------------------
if ww_have git-cliff; then
  render_with_git_cliff || ww_die "git-cliff failed to render the changelog."
else
  ww_warn "git-cliff not found — using the dependency-free git-log fallback (install git-cliff for richer output)."
  render_with_git_log
fi

ww_is_dry_run && exit 0
[ -s "${CHANGELOG_FILE}" ] || ww_die "changelog generation produced no output (${CHANGELOG_FILE})."
ww_log "changelog up to date: ${CHANGELOG_FILE}"
