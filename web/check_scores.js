// node check_scores.js [episodes] [temperature]   scores every driver in the JavaScript simulator on freshly scrambled arms
const path = require('path');
const { Sim, scrambledArm, EPISODE_STEPS, TRIAL_STEPS } = require('./sim.js');
const { ArmAgent } = require('./agent.js');
const episodes = +process.argv[2] || 40, temperature = process.argv[3] === undefined ? 1 : +process.argv[3];
for (const [wiring, name] of [['motors', 'motors3']]) {
  require(path.join(__dirname, 'weights', name + '.js'));
  const agent = new ArmAgent(globalThis.ARM_AGENTS[name]);
  const line = [];
  for (const pilot of ['random', 'factory', 'calibrate', 'told', 'learned']) {
    let total = 0; const n = pilot === 'learned' ? episodes : 300;
    for (let e = 0; e < n; e++) {
      const sim = new Sim('targets', 'HI', wiring); sim.arm = scrambledArm(wiring); sim.reset(); agent.reset();
      while (!sim.done) {
        let a;
        if (pilot === 'learned') { const out = agent.act(sim.obs(), sim.trialT, TRIAL_STEPS); a = agent.sample(out.logits, temperature); }
        else a = pilot === 'random' ? Math.floor(Math.random() * 12) : sim.choose(pilot);
        sim.step(a);
        if (pilot === 'learned') agent.remember(a, sim.lastStep.obs, sim.lastStep.settled, sim.lastStep.reward);
      }
      total += sim.reached;
    }
    line.push(`${pilot} ${(total / n).toFixed(2)}`);
  }
  console.log(`${wiring}: ${line.join(' | ')}   (PyTorch held-out for the learned agent: ${globalThis.ARM_AGENTS[name].held_out_targets})`);
}
