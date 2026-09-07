#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER="$script_dir/tauri-dev-launcher.mjs"
export CARGO_TARGET_X86_64_APPLE_DARWIN_RUNNER="$script_dir/tauri-dev-launcher.mjs"
exec cargo "$@"
