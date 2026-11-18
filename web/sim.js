/* The arm simulator. Same rules and numbers as python/arm_env.py, plus a writing mode for the page.
 *
 * Two kinds of controls:
 *   motors: 3 motors x 2 directions x 2 step sizes. Hidden: which joint each motor turns, and how strongly.
 *   board:  12 buttons. Hidden: which joint each button turns and by what signed step. Some are dead.
 * Either way there are 12 actions, and the observation never shows the wiring.
 */
(function (root) {
  'use strict';
  const N = 3, N_ACTIONS = 12, STEP_SIZES = [0.07, 0.21], DRAW_STEPS = [0.012, 0.06], DRAW_BOARD_SCALE = 0.3,
        LIMIT = 2.6, REACH = 0.06, DRAW_REACH = 0.022, EPISODE_STEPS = 150, TRIAL_STEPS = 20, TRAVEL_STEPS = 120, SPACING = 0.035,
        WRITE_SPACING = 0.08, WRITE_TRAVEL = 40;
  // single-stroke letters on a 4 x 6 grid: strokes separated by |, points as xy
  const FONT = {
    A: '00,26,40|13,33', B: '00,06,36,45,44,33,03|33,42,41,30,00', C: '45,36,16,05,01,10,30,41', D: '00,06,26,45,41,20,00',
    E: '40,00,06,46|03,33', F: '00,06,46|03,33', G: '45,36,16,05,01,10,30,41,43,23', H: '00,06|46,40|03,43', I: '10,30|20,26|16,36',
    J: '01,10,30,41,46', K: '00,06|46,03,40', L: '06,00,40', M: '00,06,23,46,40', N: '00,06,40,46', O: '10,01,05,16,36,45,41,30,10',
    P: '00,06,36,45,44,33,03', Q: '10,01,05,16,36,45,41,30,10|22,40', R: '00,06,36,45,44,33,03|23,40', S: '45,36,16,05,04,13,33,42,41,30,10,01',
    T: '06,46|26,20', U: '06,01,10,30,41,46', V: '06,20,46', W: '06,10,23,30,46', X: '00,46|06,40', Y: '06,23,46|23,20', Z: '06,46,00,40',
    0: '10,01,05,16,36,45,41,30,10', 1: '15,26,20|10,30', 2: '05,16,36,45,44,00,40', 3: '05,16,36,45,44,33,13|33,42,41,30,10,01', 4: '30,36,02,42',
    5: '46,06,03,33,42,41,30,10,01', 6: '45,36,16,05,01,10,30,41,42,33,03', 7: '06,46,20', 8: '13,04,05,16,36,45,44,33,13,02,01,10,30,41,42,33',
    9: '01,10,30,41,45,36,16,05,04,13,43' };
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  function penOf(theta, lengths) {
    let c = 0, x = 0, y = 0;
    for (let i = 0; i < N; i++) { c += theta[i]; x += lengths[i] * Math.cos(c); y += lengths[i] * Math.sin(c); }
    return [x, y];
  }
  /* Joint angles after action a. Motors: motor = a >> 2, + when a % 4 < 2, big step when a is odd.
     fine = the writing mode's smaller steps. */
  function gauss() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
  /* live = the real move, where a jittery motor adds random error. Predictions (live off) stay exact. */
  function effectOf(a, theta, arm, fine, live) {
    const th = theta.slice();
    if (arm.board) th[arm.board.joint[a]] += arm.board.step[a] * (fine ? DRAW_BOARD_SCALE : 1);
    else {
      const motor = a >> 2, dir = a % 4 < 2 ? 1 : -1, size = (fine ? DRAW_STEPS : STEP_SIZES)[a % 2];
      let move = dir * size * arm.gain[motor];
      if (live && arm.jitter) move *= 1 + arm.jitter * gauss();
      th[arm.perm[motor]] += move;
      if (arm.crosstalk) th[(arm.perm[motor] + 1) % N] += arm.crosstalk * move;      // the motor also drags the next joint
    }
    for (let i = 1; i < N; i++) th[i] = clamp(th[i], -LIMIT, LIMIT);
    return th;
  }
  // The arm as it left the factory. The factory board is laid out exactly like the three motors.
  function factoryArm(wiring) {
    const arm = { lengths: [0.35, 0.35, 0.35] };
    if (wiring === 'board') arm.board = { joint: Array.from({ length: N_ACTIONS }, (_, a) => a >> 2),
                                          step: Array.from({ length: N_ACTIONS }, (_, a) => (a % 4 < 2 ? 1 : -1) * STEP_SIZES[a % 2]) };
    else { arm.perm = [0, 1, 2]; arm.gain = [1, 1, 1]; }
    return arm;
  }
  // A freshly broken arm, drawn the way training draws them.
  function scrambledArm(wiring) {
    const arm = { lengths: [0, 1, 2].map(() => rand(0.2, 0.5)) };
    if (wiring === 'board') {
      const buttons = [];
      for (let j = 0; j < N; j++) for (const sign of [1, -1]) buttons.push([j, sign * rand(0.04, 0.25)]);   // every joint can go both ways
      while (buttons.length < N_ACTIONS) buttons.push([Math.floor(Math.random() * N), Math.random() < 0.15 ? 0 : rand(0.04, 0.25) * (Math.random() < 0.5 ? -1 : 1)]);
      shuffle(buttons);
      arm.board = { joint: buttons.map(b => b[0]), step: buttons.map(b => b[1]) };
    } else {
      arm.perm = shuffle([0, 1, 2]);
      arm.gain = [0, 1, 2].map(() => rand(0.5, 1.5) * (Math.random() < 0.5 ? -1 : 1));
      if (Math.random() < 0.25) arm.gain[Math.floor(Math.random() * N)] = 0;
    }
    return arm;
  }

  class Sim {
    constructor(mode, word, wiring) { this.arm = factoryArm(wiring); this.mode = mode || 'targets'; this.word = word || 'HI'; this.reset(); }
    get wiring() { return this.arm.board ? 'board' : 'motors'; }
    get fine() { return this.mode === 'draw'; }
    /* 'draw' and 'write' both follow the points of a word. draw uses finer steps and a small touch radius (hand-written
       drivers only). write keeps the step sizes and touch radius the learned agents were trained with. */
    get scripted() { return this.mode === 'draw' || this.mode === 'write'; }
    pen() { return penOf(this.theta, this.arm.lengths); }
    get reach() { return this.mode === 'draw' ? DRAW_REACH : REACH; }
    get limit() { return this.scripted && !this.path[this.wi].down ? (this.mode === 'draw' ? TRAVEL_STEPS : WRITE_TRAVEL) : TRIAL_STEPS; }
    obs() {                                              // what an agent sees: never the wiring
      const p = this.pen(), dx = this.target[0] - p[0], dy = this.target[1] - p[1], o = new Float32Array(5 + 2 * N);
      o[0] = p[0]; o[1] = p[1]; o[2] = dx; o[3] = dy; o[4] = Math.hypot(dx, dy);
      for (let i = 0; i < N; i++) { o[5 + i] = Math.sin(this.theta[i]); o[5 + N + i] = Math.cos(this.theta[i]); }
      return o;
    }
    reset() {
      this.t = 0; this.reached = 0; this.done = false; this.hit = null; this.lastStep = null;
      this.last = new Array(N_ACTIONS).fill(null);       // what the self-calibrating autopilot remembers
      if (this.scripted) {
        this.theta = [Math.PI / 2 + rand(-0.5, 0.5), rand(-1, 1), rand(-1, 1)];
        this.layout(); this.ink = []; this.wi = 0; this.trialT = 0; this.target = this.path[0].p;
        this.trace = [this.pen()]; this.touched = this.path.map(() => false);
      } else {
        this.theta = [rand(-Math.PI, Math.PI), rand(-2.2, 2.2), rand(-2.2, 2.2)];
        this.trace = [this.pen()]; this.newTarget();
      }
    }
    layout() {                                           // place the word inside the arm's reach, then cut it into points
      const L = this.arm.lengths.reduce((a, b) => a + b, 0);
      let chars = [...this.word.toUpperCase()].filter(c => FONT[c]).slice(0, 5);
      if (!chars.length) chars = ['H', 'I'];
      const big = this.mode === 'write', spacing = big ? WRITE_SPACING : SPACING;
      const n = chars.length, h = Math.min((big ? 0.5 : 0.36) * L, (big ? 1.3 : 1.1) * L / (n * 0.667 + (n - 1) * 0.2)), w = h * 4 / 6, gap = 0.2 * h;
      const x0 = -(n * w + (n - 1) * gap) / 2, y0 = (big ? 0.28 : 0.36) * L;
      this.strokes = []; this.path = [];
      chars.forEach((c, i) => FONT[c].split('|').forEach(s =>
        this.strokes.push(s.split(',').map(q => [x0 + i * (w + gap) + (+q[0]) / 4 * w, y0 + (+q[1]) / 6 * h]))));
      for (const stroke of this.strokes) {
        this.path.push({ p: stroke[0], down: false });   // the pen travels to the start of a stroke without inking
        for (let i = 1; i < stroke.length; i++) {
          const a = stroke[i - 1], b = stroke[i], k = Math.max(1, Math.round(dist(a, b) / spacing));
          for (let j = 1; j <= k; j++) this.path.push({ p: [a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k], down: true });
        }
      }
    }
    newTarget() {                                        // the pen position of a random nearby pose, about 0.35 away
      const p = this.pen(); let best = null, bestScore = Infinity;
      for (let k = 0; k < 4; k++) {
        const th = this.theta.map((v, i) => { const x = v + rand(-1.2, 1.2); return i ? clamp(x, -LIMIT, LIMIT) : x; });
        const cand = penOf(th, this.arm.lengths), score = Math.abs(dist(cand, p) - 0.35);
        if (score < bestScore) { best = cand; bestScore = score; }
      }
      this.target = best; this.trialT = 0;
    }
    retarget() { if (this.scripted) this.reset(); else this.newTarget(); }   // after the arm's shape changed
    next() {
      this.trialT = 0;
      if (!this.scripted) return this.newTarget();
      this.wi++;
      if (this.wi >= this.path.length) { this.wi = this.path.length - 1; this.done = true; } else this.target = this.path[this.wi].p;
    }
    /* The hand-written drivers. factory: plans with the factory arm. calibrate: tries every action once,
       remembers the pen displacement each produced, then takes the best predicted step. told: reads the true wiring. */
    choose(pilot) {
      const p = this.pen(); let bestA = 0, bestD = Infinity;
      if (pilot === 'calibrate') { const untried = this.last.findIndex(v => v === null); if (untried >= 0) return untried; }
      const factory = pilot === 'factory' ? factoryArm(this.wiring) : null;
      for (let a = 0; a < N_ACTIONS; a++) {
        let landing;
        if (pilot === 'told') landing = penOf(effectOf(a, this.theta, this.arm, this.fine), this.arm.lengths);
        else if (pilot === 'factory') {
          const q0 = penOf(this.theta, factory.lengths), q1 = penOf(effectOf(a, this.theta, factory, this.fine), factory.lengths);
          landing = [p[0] + q1[0] - q0[0], p[1] + q1[1] - q0[1]];
        } else landing = [p[0] + this.last[a][0], p[1] + this.last[a][1]];
        const d = dist(landing, this.target);
        if (d < bestD) { bestD = d; bestA = a; }
      }
      return bestA;
    }
    step(a) {                                            // returns 'reached', 'timeout' or null, and sets this.done
      const p0 = this.pen(), obs0 = this.obs(), d0 = dist(p0, this.target);
      this.theta = effectOf(a, this.theta, this.arm, this.fine, true);
      const p1 = this.pen(), d1 = dist(p1, this.target), touched = d1 < this.reach;
      // what a learning agent stores: the observation after the move with the SAME target, and the reward
      this.lastStep = { obs: obs0, settled: this.obs(), reward: 2 * (d0 - d1) + (touched ? 1 : 0), action: a };
      this.last[a] = [p1[0] - p0[0], p1[1] - p0[1]];
      if (this.mode === 'draw') { if (this.path[this.wi].down) this.ink.push([p0, p1]); }
      else { this.trace.push(p1); if (this.trace.length > 400) this.trace.shift(); }
      if (this.scripted && touched) this.touched[this.wi] = true;
      this.t++; this.trialT++;
      let event = null;
      if (touched) { this.reached++; event = 'reached'; this.hit = this.target; this.next(); }
      else if (this.trialT >= this.limit) { event = 'timeout'; this.next(); }
      if (!this.scripted) this.done = this.t >= EPISODE_STEPS;
      return event;
    }
  }

  const api = { Sim, scrambledArm, factoryArm, effectOf, penOf, N, N_ACTIONS, STEP_SIZES, LIMIT, REACH, EPISODE_STEPS, TRIAL_STEPS };
  root.ArmSim = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
