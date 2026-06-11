#!/usr/bin/env bash
# WorktreeCreate hook: replaces Claude Code's default git worktree creation.
# Creates a new worktree-<name> branch from the freshly-fetched origin/main
# under .claude/worktrees/, then symlinks untracked runtime dirs back to the main
# checkout so e.g. TensorBoard logs written in the worktree persist centrally
# and survive worktree cleanup.
set -euo pipefail

NAME="$(jq -r '.name')"
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
DIR="$ROOT/.claude/worktrees/$NAME"

# Re-entering an existing worktree: just hand back its path.
if [ -d "$DIR" ]; then
  echo "$DIR"
  exit 0
fi

# Always branch from the latest upstream main: fetch it, then use origin/main.
git -C "$ROOT" fetch origin main >&2
git -C "$ROOT" worktree add -b "worktree-$NAME" "$DIR" origin/main >&2

# Symlink shared untracked dirs. logs is ensured (so links never dangle);
# the rest are linked only if they already exist in the main checkout.
mkdir -p "$ROOT/logs"
for d in logs roms videos images; do
  [ -e "$ROOT/$d" ] || continue
  ln -sfn "$ROOT/$d" "$DIR/$d"
done

# Generate compile_commands.json fresh in the worktree so clangd indexes THIS
# branch. Copying the main checkout's index is wrong: its `directory` and the
# absolute execroot paths baked into each command point at the main tree, not
# this worktree, so clangd would resolve sources against the wrong checkout
# until the first refresh overwrote it. Don't let a refresh failure abort the
# hook (set -e) before we print $DIR — a missing index is a clangd nicety, but
# no stdout means Claude Code can't locate the worktree. SessionStart's refresh
# retries when the index is missing anyway.
( cd "$DIR" && bazel run //:refresh_compile_commands >&2 ) \
  || echo "warning: refresh_compile_commands failed; clangd index not seeded" >&2

echo "$DIR"
