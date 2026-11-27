// Fast exact makefun-diff 1.0.0 driver: tunes V8 flags for single-CPU cold execution,
// reads /input/cases.json directly into WASM linear memory, and writes /output/answers.json.
require('node:v8').setFlagsFromString(
  '--no-incremental-marking --no-wasm-tier-up --no-concurrent-recompilation --no-concurrent-marking --no-parallel-scavenge'
);

const { runCasesFile } = require('./engine.cjs');

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

runCasesFile(INPUT_DIR + '/cases.json', OUTPUT_DIR + '/answers.json');
