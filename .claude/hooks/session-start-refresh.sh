#!/usr/bin/env bash
# SessionStart hook: the PostToolUse refresh only sees Claude's own Write/Edit
# calls, so the index goes stale when files arrive another way between sessions
# — git merge/pull/checkout/rebase, or edits made in the user's editor. At
# session start, refresh Hedron's compile_commands.json, but only when the build
# graph or a C/C++ source is actually newer than the index; otherwise no-op so
# warm sessions start instantly. A missing index always refreshes. (Cold-cache
# refreshes run a full build and can be slow; warm ones are just re-extraction.)
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}"

# Linked worktrees need the gitignored runtime dirs symlinked from the main
# checkout — without roms/, the //:roms glob in BUILD hard-fails package
# loading, breaking the refresh below. worktree-create.sh seeds these links,
# but worktrees made by other means (e.g. Claude Code's built-in worktree
# creation) bypass that hook, so self-heal here. -e is false for a dangling
# link, which ln -sfn then replaces; an existing real dir is left alone.
main="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
if [ "$main" != "$PWD" ]; then
  for d in logs roms videos images; do
    if [ -e "$main/$d" ] && [ ! -e "$d" ]; then
      ln -sfn "$main/$d" "$d"
    fi
  done
fi

cc="compile_commands.json"

need=0
if [ ! -f "$cc" ]; then
  need=1
# Prune generated/symlinked trees, then flag a refresh if any build file or
# C/C++ source/header is newer than the index. -print -quit stops at the first
# hit; grep turns "any output" into the exit status. (*.BUILD covers the
# external-repo build files ale.BUILD/libtorch.BUILD whose flags reach us.)
elif find . \
       \( -path ./.git -o -path './bazel-*' -o -path ./external \) -prune -o \
       \( -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.cu' \
          -o -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.cuh' \
          -o -name '*.bzl' -o -name '*.bazel' -o -name '*.BUILD' -o -name 'BUILD' \) \
       -newer "$cc" -print -quit | grep -q .; then
  need=1
fi

[ "$need" -eq 1 ] || exit 0

# Mirror refresh-compile-commands.sh: on failure, surface the error as context
# rather than blocking — a stale index is a warning, not a hard stop.
if out="$(bazel run //:refresh_compile_commands 2>&1)"; then
  printf '{"suppressOutput": true}'
else
  jq -n --arg ctx "SessionStart refresh_compile_commands failed; clangd diagnostics may be stale until fixed:
$out" '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $ctx}}'
fi
