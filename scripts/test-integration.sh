#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
environment_file="$repository_root/.env.integration"

if [ ! -f "$environment_file" ]; then
    echo "Copy .env.integration.example to .env.integration and configure caller-owned paths." >&2
    exit 1
fi

set -a
. "$environment_file"
set +a

: "${NIXE_REAL_PACKAGE:?set NIXE_REAL_PACKAGE in .env.integration}"
: "${NIXE_KEYS_DIR:?set NIXE_KEYS_DIR in .env.integration}"

NIXE_REAL_NSP=${NIXE_REAL_NSP:-$NIXE_REAL_PACKAGE}
NIXE_TEST_NSP=${NIXE_TEST_NSP:-$NIXE_REAL_NSP}
NIXE_PROD_KEYS=${NIXE_PROD_KEYS:-$NIXE_KEYS_DIR/prod.keys}
if [ -z "${NIXE_TITLE_KEYS:-}" ] && [ -f "$NIXE_KEYS_DIR/title.keys" ]; then
    NIXE_TITLE_KEYS="$NIXE_KEYS_DIR/title.keys"
fi
export NIXE_REAL_NSP NIXE_TEST_NSP NIXE_PROD_KEYS NIXE_TITLE_KEYS

cd "$repository_root"
cargo test -p nixe-loader-content --test external_nca -- --ignored
cargo test -p nixe-loader-executable --test real_npdm -- --ignored
cargo test -p nixe-loader-executable --test real_nso -- --ignored
cargo test -p nixe-runtime --test real_prepared_module_memory -- --ignored
cargo test -p nixe-runtime --test real_launch_plan -- --ignored
