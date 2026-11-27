const e = process.env;
const inFile = (e.MAKEFUN_INPUT_DIR || '/input') + '/cases.json';
const outFile = (e.MAKEFUN_OUTPUT_DIR || '/output') + '/answers.json';

let done = false;
if (process.platform === 'linux' && process.arch === 'x64') {
  try {
    process.dlopen(module, __dirname + '/diff.node');
    done = module.exports.runFile(inFile, outFile, 1) === 0;
  } catch {}
}

if (!done) {
  require('node:v8').setFlagsFromString(
    '--no-incremental-marking --no-wasm-tier-up --no-concurrent-recompilation --no-concurrent-marking --no-parallel-scavenge'
  );
  require('./engine.cjs').runCasesFile(inFile, outFile);
}
