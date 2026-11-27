// Exact makefun-diff 1.0.0 driver:
// Fast path on Linux x86_64 uses zero-libc AVX2/SSE2 N-API addon (diff.node) with zero require() overhead;
// automatically falls back to SIMD128 WebAssembly + pure-JS engine (engine.cjs) on other platforms or non-canonical JSON.
const inFile = (process.env.MAKEFUN_INPUT_DIR || '/input') + '/cases.json';
const outFile = (process.env.MAKEFUN_OUTPUT_DIR || '/output') + '/answers.json';

let done = false;
if (process.platform === 'linux' && process.arch === 'x64') {
  try {
    process.dlopen(module, __dirname + '/diff.node');
    done = module.exports.runFile(inFile, outFile) === 0;
  } catch {}
}

if (!done) {
  require('node:v8').setFlagsFromString(
    '--no-incremental-marking --no-wasm-tier-up --no-concurrent-recompilation --no-concurrent-marking --no-parallel-scavenge'
  );
  require('./engine.cjs').runCasesFile(inFile, outFile);
}
