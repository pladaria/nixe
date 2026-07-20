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

: "${SWIITX_REAL_PACKAGE:?set SWIITX_REAL_PACKAGE in .env.integration}"
: "${SWIITX_KEYS_DIR:?set SWIITX_KEYS_DIR in .env.integration}"

SWIITX_REAL_NSP=${SWIITX_REAL_NSP:-$SWIITX_REAL_PACKAGE}
SWIITX_TEST_NSP=${SWIITX_TEST_NSP:-$SWIITX_REAL_NSP}
SWIITX_PROD_KEYS=${SWIITX_PROD_KEYS:-$SWIITX_KEYS_DIR/prod.keys}
if [ -z "${SWIITX_TITLE_KEYS:-}" ] && [ -f "$SWIITX_KEYS_DIR/title.keys" ]; then
    SWIITX_TITLE_KEYS="$SWIITX_KEYS_DIR/title.keys"
fi
export SWIITX_REAL_NSP SWIITX_TEST_NSP SWIITX_PROD_KEYS SWIITX_TITLE_KEYS

cd "$repository_root"
cargo test -p swiitx-loader-content --test external_nca -- --ignored
cargo test -p swiitx-loader-executable --test real_npdm -- --ignored
cargo test -p swiitx-loader-executable --test real_nso -- --ignored
cargo test -p swiitx-runtime --test real_prepared_module_memory -- --ignored
cargo test -p swiitx-runtime --test real_launch_plan -- --ignored
