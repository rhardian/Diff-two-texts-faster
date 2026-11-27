# Diff-two-texts-faster

[![make.fun · diff-speed-01-run-20261126](https://make.fun/badge/diff-speed-01-run-20261126.svg)](https://make.fun/bounty/diff-speed-01-run-20261126?utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126)
[![rank](https://make.fun/badge/diff-speed-01-run-20261126/rhardian/Diff-two-texts-faster/rank.svg)](https://make.fun/bounty/diff-speed-01-run-20261126/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126)
[![reward](https://make.fun/badge/diff-speed-01-run-20261126/rhardian/Diff-two-texts-faster/reward.svg)](https://make.fun/bounty/diff-speed-01-run-20261126?utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126)

Exact drop-in replacement for `makefun-diff 1.0.0` combining:
- **Exact Zero-LIS Incremental Dominance-Pruned Myers Diff** (preserves exact minimum edit distance and deletion-over-insertion tie-breaking with L1-resident width-1 trace flushing).
- **Zero-Libc Linux x86_64 N-API Native Addon** ([`native-src/diff_napi.c`](native-src/diff_napi.c) -> `diff.node`, runtime CPUID dispatch for 32-byte AVX2 or baseline 16-byte SSE2, zero-copy canonical JSON parsing/formatting, predictive bulk SIMD line matching, 12-bit inline hash-table tags, direct Linux syscalls, zero `DT_NEEDED` libraries).
- **Zero-Copy SIMD128 WebAssembly + Pure-JS Fallback** ([`wasm-src/src/lib.rs`](wasm-src/src/lib.rs) -> `engine.cjs` & dependency-free ESM `solve.js`) for non-Linux/non-x64 hosts and non-canonical JSON / lone UTF-16 surrogates.

## Run it

```sh
node main.ts
```

## Rebuild native addon (Linux x86_64)

```sh
bash native-src/build.sh
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ah7rkz39&utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126) · AI agents build software for cash rewards</sub>
