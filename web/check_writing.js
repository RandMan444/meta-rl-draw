// node check_writing.js [words] ['[[strength, drag], ...]']
// The page's writing task without a browser: the share of a word's points each driver inks, starting from the factory
// arm with the motors scaled by `strength` and each motor dragging the next joint by `drag` times its move.
const path = require('path');
const { Sim, factoryArm, TRIAL_STEPS, EPISODE_STEPS } = require('./sim.js');
const { ArmAgent } = require('./agent.js'); const { RecurrentAgent } = require('./recurrent.js');
require(path.join(__dirname, 'weights', 'motors3.js')); require(path.join(__dirname, 'weights', 'recurrent3.js'));
const words = +process.argv[2] || 8, WORD = 'HELLO', TEMPERATURE = 0.25, MAX_MOVES = 2600;
const cases = process.argv[3] ? JSON.parse(process.argv[3])
  : [[1, 0], [0.5, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1], [2, 0.5], [2, 1]];
function write(pilot, strength, drag) {
  const sim = new Sim('write', WORD, 'motors'), arm = factoryArm('motors');
  arm.gain = arm.gain.map(g => g * strength); arm.crosstalk = drag; sim.arm = arm; sim.reset();
  const agent = pilot === 'learned' ? new ArmAgent(globalThis.ARM_AGENTS.motors3) : pilot === 'recurrent' ? new RecurrentAgent(globalThis.ARM_RIVALS.recurrent3) : null;
  let moves = 0;
  while (!sim.done && moves < MAX_MOVES) {
    if (agent && moves && moves % EPISODE_STEPS === 0) agent.reset();          // both agents were trained on 150-move episodes
    const a = agent ? agent.sample(agent.act(sim.obs(), Math.min(sim.trialT, TRIAL_STEPS), TRIAL_STEPS).logits, TEMPERATURE) : sim.choose(pilot);
    sim.step(a); moves++;
    if (agent) agent.remember(a, sim.lastStep.obs, sim.lastStep.settled, sim.lastStep.reward);
  }
  const total = sim.path.filter(p => p.down).length, inked = sim.path.reduce((n, p, i) => n + (p.down && sim.touched[i] ? 1 : 0), 0);
  return inked / total;
}
for (const [strength, drag] of cases) {
  const line = ['learned', 'recurrent', 'calibrate', 'factory'].map(pilot => {
    const rs = Array.from({ length: words }, () => write(pilot, strength, drag));
    return `${pilot} ${(100 * rs.reduce((a, b) => a + b, 0) / words).toFixed(0)}% (worst word ${Math.round(100 * Math.min(...rs))}%)`;
  });
  console.log(`strength ${strength}x, drag ${Math.round(100 * drag)}%: ${line.join(' | ')}`);
}
