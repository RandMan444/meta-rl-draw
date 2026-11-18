// node check_unseen.js [arms]   unseen damage in the JavaScript simulator: learned agent and recurrent baseline (both at temperature 0.25) against the self-calibrating autopilot
const path = require('path');
const { Sim, scrambledArm, TRIAL_STEPS } = require('./sim.js');
const { ArmAgent } = require('./agent.js');
require(path.join(__dirname, 'weights', 'motors3.js')); require(path.join(__dirname, 'weights', 'recurrent3.js'));
const { RecurrentAgent } = require('./recurrent.js');
const learned = new ArmAgent(globalThis.ARM_AGENTS.motors3), recurrent = new RecurrentAgent(globalThis.ARM_RIVALS.recurrent3), arms = +process.argv[2] || 60;
const CASES = [['trained range', a => a], ['every motor 2x stronger', a => { a.gain = [2, -2, 2]; return a; }], ['every motor 3x stronger', a => { a.gain = [3, -3, 3]; return a; }],
  ['crosstalk 1.0', a => { a.crosstalk = 1; return a; }], ['jitter 0.6', a => { a.jitter = 0.6; return a; }], ['quarter strength', a => { a.gain = [0.25, -0.25, 0.25]; return a; }]];
for (const [name, damage] of CASES) {
  const score = {};
  for (const pilot of ['calibrate', 'recurrent', 'learned']) {
    const agent = pilot === 'learned' ? learned : pilot === 'recurrent' ? recurrent : null;
    let total = 0;
    for (let e = 0; e < arms; e++) {
      const sim = new Sim('targets', 'HI', 'motors'); sim.arm = damage(scrambledArm('motors')); sim.reset(); if (agent) agent.reset();
      while (!sim.done) {
        let a;
        if (agent) a = agent.sample(agent.act(sim.obs(), sim.trialT, TRIAL_STEPS).logits, 0.25); else a = sim.choose(pilot);
        sim.step(a);
        if (agent) agent.remember(a, sim.lastStep.obs, sim.lastStep.settled, sim.lastStep.reward);
      }
      total += sim.reached;
    }
    score[pilot] = (total / arms).toFixed(2);
  }
  console.log(`${name}: learned ${score.learned} | recurrent ${score.recurrent} | self-calibrating ${score.calibrate}`);
}
