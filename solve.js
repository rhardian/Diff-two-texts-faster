// Dependency-free ES module drop-in for makefun-diff 1.0.0.
import engine from './engine.cjs';

export const FORMAT = 'makefun-diff/1';
export const CONTEXT = 3;

export function run(input) {
  return engine.run(input);
}

export default { run, FORMAT, CONTEXT };
