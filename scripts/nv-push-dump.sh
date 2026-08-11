#!/usr/bin/env bash
set -euo pipefail

readonly repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly docker_context="$repository_root/docker/nv-push-dump"
readonly image="nixe/nv-push-dump:mesa-26.0.6"

usage() {
    echo "Usage: $0 PUSHBUFFER.bin [ARCHITECTURE]" >&2
}

if (( $# < 1 || $# > 2 )); then
    usage
    exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to run nv_push_dump" >&2
    exit 1
fi

readonly requested_path="$1"
readonly architecture="${2:-MAXWELL}"

if [[ ! -f "$requested_path" ]]; then
    echo "pushbuffer dump does not exist: $requested_path" >&2
    exit 1
fi

readonly input_directory="$(cd -- "$(dirname -- "$requested_path")" && pwd)"
readonly input_filename="$(basename -- "$requested_path")"

if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Building $image..." >&2
    docker build --tag "$image" "$docker_context"
fi

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "$input_directory:/input:ro" \
    "$image" "/input/$input_filename" "$architecture"
