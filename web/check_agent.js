// node check_agent.js <name>   replays the PyTorch trace through agent.js and reports the largest logit difference
const fs = require('fs'), path = require('path');
const name = process.argv[2] || 'motors3';
require(path.join(__dirname, 'weights', name + '.js'));
const { ArmAgent } = require('./agent.js');
const trace = JSON.parse(fs.readFileSync(path.join(__dirname, 'weights', name + '.trace.json')));
const agent = new ArmAgent(globalThis.ARM_AGENTS[name]);
let worst = 0, worstV = 0, codeMismatch = 0, t0 = Date.now();
for (const s of trace.steps) {
  const out = agent.act(Float32Array.from(s.obs), s.trial_t, 20);
  s.logits.forEach((v, a) => { worst = Math.max(worst, Math.abs(v - out.logits[a])); });
  worstV = Math.max(worstV, Math.abs(s.value - out.value));
  s.codes.forEach((c, a) => { if (c !== out.codes[a]) codeMismatch++; });
  agent.remember(s.action, Float32Array.from(s.obs), Float32Array.from(s.settled), s.reward);
}
console.log(`${name}: ${trace.steps.length} steps, max |logit difference| ${worst.toExponential(2)}, max |value difference| ${worstV.toExponential(2)}, code mismatches ${codeMismatch}, ${((Date.now() - t0) / trace.steps.length).toFixed(1)} ms per step`);
process.exit(worst < 1e-3 ? 0 : 1);
