set -e

workspace=$(bazel info workspace)

tmp_directory=/tmp/ale-bazel-flamegraph
perf_file_path=$tmp_directory/perf.data
svg_file_path=$tmp_directory/perf.svg

# -fno-omit-frame-pointer is advised from http://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html#Instructions.
bazel build \
  --compilation_mode=dbg \
  --copt=-fno-omit-frame-pointer \
  --copt=-g \
  --linkopt=-fno-omit-frame-pointer \
  --strip=never \
  //src/bin:ale_ppo_threaded

mkdir -p $tmp_directory
echo "Require sudo to run perf:"
trap 'echo -e "\nProgram stopped by user. Continuing with flamegraph generation..."' SIGINT
# bazel-bin/src/bin/ale_ppo_threaded.runfiles/_main/src/bin/ale_ppo_threaded
sudo perf record -a -b -g --output=$perf_file_path $workspace/bazel-bin/src/bin/ale_ppo_threaded $workspace/roms/breakout.bin $workspace/logs/perf $workspace/images/perf || true
trap - SIGINT
sudo chown $USER $perf_file_path

perf script --input=$perf_file_path | $workspace/third_party/FlameGraph/stackcollapse-perf.pl | $workspace/third_party/FlameGraph/flamegraph.pl > $svg_file_path

if [ -x "$(command -v google-chrome)" ]; then
  google-chrome $svg_file_path
fi
echo "Perf data recorded at $perf_file_path"
echo "Flamegraph generated at $svg_file_path"