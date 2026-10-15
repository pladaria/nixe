#!/usr/bin/env bash
# Cargo's Linux runner; preserve Cargo's own dynamic-library search paths.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARGUMENTS...]" >&2
    exit 2
fi
repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
loader_lib="$repository_root/dev-tools/vulkan-loader/install/lib"
if [[ -f $loader_lib/libvulkan.so.1 ]]; then
    export LD_LIBRARY_PATH="$loader_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$@"
