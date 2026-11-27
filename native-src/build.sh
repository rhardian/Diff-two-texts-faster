#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -DBF_SHIFT=18 -DSHORT_HASH_MIN=12 -shared -fPIC -nostdlib -fno-builtin \
    -fno-stack-protector -fno-asynchronous-unwind-tables -fno-ident \
    -fvisibility=hidden -ffunction-sections -fdata-sections \
    -Wl,--gc-sections,-s,--build-id=none,-z,noseparate-code,-z,max-page-size=4096,--allow-shlib-undefined \
    -o ../d.node diff_napi.c
ls -lh ../d.node
