#!/usr/bin/env bash
# PostToolUse hook (Write|Edit): keep Hedron's compile_commands.json fresh so
# clangd stops emitting false positives. Refreshes only on the changes that
# actually invalidate the index — a newly written C/C++ source/header, or any
# edit to a BUILD/.bzl/MODULE.bazel file — and skips routine body edits to
# files clangd already knows. Blocking by design: the refresh finishes before
# the next diagnostics read. (Bazel 9 / bzlmod: MODULE.bazel is caught by the
# *.bazel glob; there is no WORKSPACE.)
set -euo pipefail

input="$(cat)"
tool="$(jq -r '.tool_name // ""' <<<"$input")"
file="$(jq -r '.tool_input.file_path // .tool_response.filePath // ""' <<<"$input")"

refresh=0
case "$file" in
  # Build graph changed -> flags for many files may be stale.
  # (*.bazel covers BUILD.bazel and MODULE.bazel.)
  *.bzl|*.bazel|*/BUILD|BUILD) refresh=1 ;;
  # New C/C++ file (Write) is absent from compile_commands.json until refreshed.
  # An Edit only touches an already-indexed body, so clangd copes without one.
  *.cc|*.cpp|*.cxx|*.cu|*.h|*.hpp|*.hh|*.cuh)
    [ "$tool" = "Write" ] && refresh=1 ;;
esac

[ "$refresh" -eq 1 ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}"

# On failure, hand the error back to Claude as context instead of blocking the
# turn — the edit already landed; a stale index is a warning, not a hard stop.
if out="$(bazel run //:refresh_compile_commands 2>&1)"; then
  printf '{"suppressOutput": true}'
else
  jq -n --arg ctx "refresh_compile_commands failed; clangd diagnostics may be stale until fixed:
$out" '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
fi
