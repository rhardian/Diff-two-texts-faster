const e = process.env;
if (e.MAKEFUN_INPUT_DIR || e.MAKEFUN_OUTPUT_DIR) module.exports = { i: (e.MAKEFUN_INPUT_DIR || '/input') + '/cases.json', o: (e.MAKEFUN_OUTPUT_DIR || '/output') + '/answers.json' };
try { process.dlopen(module, __dirname + '/d.node'); } catch {}
require('./engine.cjs').runCasesFile((e.MAKEFUN_INPUT_DIR || '/input') + '/cases.json', (e.MAKEFUN_OUTPUT_DIR || '/output') + '/answers.json');
