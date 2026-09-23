#!/usr/bin/env bash
# Copy working sources, homebrews and keys to the native Arm host.
# Includes uncommitted source changes; excludes all non-homebrew ROMs.
# Local sources are authoritative: obsolete remote sources/homebrews are deleted.
# Excluded files and remote-only keys are preserved. Preview with --dry-run.
# Do not keep independent source edits in these remote working directories.
set -euo pipefail

case "${1:-}" in
    '') dry_run=() ;;
    --dry-run) dry_run=(--dry-run) ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

nixe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
wasmtime_root=${WASMTIME_ROOT:-"$(dirname -- "$nixe_root")/wasmtime"}
test -f "$nixe_root/Cargo.toml"
test -f "$wasmtime_root/cranelift/codegen/Cargo.toml"
test -d "$nixe_root/keys"

# Exclusions also protect receiver files: do not add --delete-excluded.
common=(
    --archive --compress --stats --delete-delay --itemize-changes
    --rsh='ssh -o BatchMode=yes'
    --exclude=.git --exclude=target --exclude=node_modules
    --exclude=.agents --exclude=.codex --exclude=.claude
    --exclude=.idea --exclude=.vscode --exclude=.vs
    --exclude=__pycache__ --exclude=.mypy_cache --exclude=.cache
    --exclude='.env' --exclude='.env.*'
    --exclude='*.profraw' --exclude='*.profdata' --exclude='perf.data*'
    --exclude='*.coredump' --exclude='*.rs.bk' --exclude='*.pyc'
    --exclude='*.swp' --exclude='*.swo' --exclude='*~'
    --exclude=.DS_Store --exclude=Thumbs.db
)

# Example environment files are source, not private machine configuration.
common=(--include='.env*.example' "${common[@]}")

if (( ${#dry_run[@]} == 0 )); then
    ssh -o BatchMode=yes pi5 \
        'mkdir -p projects/nixe projects/wasmtime && install -d -m 700 projects/nixe/keys'
fi

rsync "${common[@]}" "${dry_run[@]}" \
    --include=/roms/ --include='/roms/homebrew/***' --exclude='/roms/*' \
    --exclude=/keys/ --exclude=/notes/ \
    --exclude=/cache/ --exclude=/storage/ --exclude=/dump/ \
    --exclude='/perf.*' --exclude='/perf-*' \
    --exclude='/docker/reference-*' \
    --exclude=/fuzz/artifacts/ --exclude=/fuzz/corpus/ \
    "$nixe_root/" pi5:projects/nixe/

rsync "${common[@]}" "${dry_run[@]}" \
    --exclude=/.cargo/ --exclude=/docs/_build/ --exclude=/docs/book/ \
    --exclude=/examples/build/ --exclude=/crates/c-api/build/ \
    --exclude=/artifacts/ --exclude=/report/ --exclude=/miri-wast/ \
    --exclude=/publish/ --exclude=/vendor/ \
    --exclude=/cranelift/isle/veri/cache/ --exclude='cranelift.dbg*' \
    "$wasmtime_root/" pi5:projects/wasmtime/

# Explicitly requested private data: no deletion or filename/content logging.
rsync --recursive --times --perms --compress --stats \
    --rsh='ssh -o BatchMode=yes' \
    --chmod=Du=rwx,Dgo=,Fu=rw,Fgo= "${dry_run[@]}" \
    "$nixe_root/keys/" pi5:projects/nixe/keys/
