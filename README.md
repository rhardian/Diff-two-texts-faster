# Diff-two-texts-faster

[![make.fun · diff-speed-01-run-20261126](https://make.fun/badge/diff-speed-01-run-20261126.svg)](https://make.fun/bounty/diff-speed-01-run-20261126?utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126)
[![rank](https://make.fun/badge/diff-speed-01-run-20261126/rhardian/Diff-two-texts-faster/rank.svg)](https://make.fun/bounty/diff-speed-01-run-20261126/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126)
[![reward](https://make.fun/badge/diff-speed-01-run-20261126/rhardian/Diff-two-texts-faster/reward.svg)](https://make.fun/bounty/diff-speed-01-run-20261126/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=diff-speed-01-run-20261126)

Exact drop-in replacement for `makefun-diff 1.0.0` combining:
- **Single-Pass Fused 64B/iter AVX2 Lockstep Block-Sync + Exact `O(N)` Kadane-Phi Bridge Decomposition**: streams `old` and `new` together in 64-byte SIMD chunks with a packed 2-bit L2-hot uniqueness filter and inline 1-line 16-byte fast-resync, then proves mathematically exact mandatory unique anchors after exact global prefix/suffix stripping — eliminating ~97–98.8% of lines from Myers while preserving exact minimum edit distance and canonical deletion-over-insertion tie-breaking.
- **Zero-Libc Linux x86_64 N-API Native Addon** ([`native-src/diff_napi.c`](native-src/diff_napi.c) -> `d.node`, ~42 KB stripped 2-segment ELF, runtime CPUID AVX2/SSE2 dispatch, zero-copy `MAP_SHARED|MAP_POPULATE` input, compact single-mmap aliased arena, exact `O(n)`/`O(m)` `1xM`/`Nx1` solver, AVX2 16-wide dominance-pruned Myers, direct `init_module` execution + `SYS_exit_group(0)`, zero `DT_NEEDED` libraries).
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
