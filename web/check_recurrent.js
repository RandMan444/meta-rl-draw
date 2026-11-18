// node check_recurrent.js [name]   replays the PyTorch trace through recurrent.js and reports the largest logit difference
const fs = require('fs'), path = require('path');
const name = process.argv[2] || 'recurrent3';
require(path.join(__dirname, 'weights', name + '.js'));
const { RecurrentAgent } = require('./recurrent.js');
const trace = JSON.parse(fs.readFileSync(path.join(__dirname, 'weights', name + '.trace.json')));
const agent = new RecurrentAgent(globalThis.ARM_RIVALS[name]);
let worst = 0;
for (const s of trace.steps) {
  const out = agent.act(Float32Array.from(s.obs), s.trial_t, 20);
  s.logits.forEach((v, a) => { worst = Math.max(worst, Math.abs(v - out.logits[a])); });
  agent.remember(s.action, null, null, s.reward);
}
console.log(`${name}: ${trace.steps.length} steps, max |logit difference| ${worst.toExponential(2)}`);
process.exit(worst < 1e-3 ? 0 : 1);
