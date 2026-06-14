#!/usr/bin/env bash
# Regenerate ALE's PGO profiles: build an instrumented train binary, run it
# briefly on a real workload, harvest the profiles into third_party/ale-pgo/,
# and rebuild with feedback. Afterwards always build/run with
# --config=ale-pgo-use to keep the optimized emulator.
# Re-run after bumping ALE or GCC (a stale profile only warns, it doesn't fail).
set -euo pipefail

workspace=$(bazel info workspace)
rom=${ROM:-$workspace/roms/breakout.bin}
config=${CONFIG:-$workspace/configs/v0.yaml}
profile_seconds=${PROFILE_SECONDS:-90}
profile_tmp=/tmp/ale-bazel-pgo
profile_dir=$workspace/third_party/ale-pgo

bazel build --compilation_mode=opt --config=ale-pgo-gen //src/bin:train

# Clean slate: .gcda from an older binary would fail the checksum merge.
rm -rf "$profile_tmp"
mkdir -p "$profile_tmp"

# Profiles are only written on a clean exit, and the stop flag is only
# checked at rollout boundaries, so: wait until rollouts actually start
# (first-run libtorch/CUDA startup can take minutes), give the run its
# profiling window, then SIGINT and wait for the graceful shutdown.
run_log=/tmp/ale-pgo-run/train.log
mkdir -p /tmp/ale-pgo-run
"$workspace/bazel-bin/src/bin/train" \
  --rom "$rom" \
  --config "$config" \
  --log-path /tmp/ale-pgo-run/logs/train \
  --video-dir /tmp/ale-pgo-run/videos \
  --group pgo-profile >"$run_log" 2>&1 &
train_pid=$!

for _ in $(seq 1 300); do
  grep -q "Rollout" "$run_log" 2>/dev/null && break
  kill -0 "$train_pid" 2>/dev/null || break
  sleep 1
done
if ! kill -0 "$train_pid" 2>/dev/null; then
  echo "error: train exited during startup; see $run_log" >&2
  exit 1
fi

echo "Training started; profiling for ${profile_seconds}s..."
sleep "$profile_seconds"
kill -INT "$train_pid"
wait "$train_pid" || true

if ! find "$profile_tmp" -name '*.gcda' -print -quit | grep -q .; then
  echo "error: no .gcda profiles in $profile_tmp; see $run_log" >&2
  exit 1
fi

rm -rf "$profile_dir"
mkdir -p "$profile_dir"
cp -a "$profile_tmp/." "$profile_dir/"
echo "Profiles harvested to $profile_dir"

bazel build --compilation_mode=opt --config=ale-pgo-use //src/bin:train
echo "Done. Use '--config=ale-pgo-use' on future builds/runs to keep PGO."
