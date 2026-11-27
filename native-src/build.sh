#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -shared -fPIC -nostdlib -fno-builtin -fno-stack-protector \
    -fno-asynchronous-unwind-tables -fvisibility=hidden \
    -Wl,--gc-sections,-s,-z,max-page-size=4096,--allow-shlib-undefined \
    -o ../diff.node diff_napi.c
readelf -d ../diff.node | grep -E "NEEDED|INIT|RPATH|RUNPATH" || true
nm -D -u ../diff.node
