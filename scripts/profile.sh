#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
profiling_directory="$repository_root/target/profiling"

cd "$repository_root"
exec sudo env "LD_LIBRARY_PATH=$profiling_directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    perf record --no-buildid -F 99 --call-graph dwarf,4096 -o perf.data -- \
    timeout -s INT -k 5s 30s "$profiling_directory/nixe-cli" --headless run "$@"
