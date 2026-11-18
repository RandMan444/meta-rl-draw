/* The page: one arm writes a word, the visitor breaks it, the learned agent copes without retraining.
   Each letter is cut into points; a piece of the letter is inked when the pen touches its point.
   Two buttons rewire the arm (damage the agents practiced on); two sliders push it past anything they practiced on. */
(function () {
  'use strict';
  const { Sim, factoryArm, N, TRIAL_STEPS, EPISODE_STEPS } = window.ArmSim;
  const $ = id => document.getElementById(id);
  const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TEMPERATURE = 0.25;        // more decisive than the training temperature of 1
  const MOVES_PER_SECOND = 45, MAX_MOVES = 2600, HOLD_MS = 2600;
  const PERMS = [[1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]];
  const DAMAGE = [['swap', 'Swap the motor wires'], ['reverse', 'Reverse a motor']];
  const TRAINED_STRENGTH = [0.5, 1.5];        // the agents practiced on motors this strong, and with no drag at all

  const on = { swap: false, reverse: false };
  let strength = 1, drag = 0;
  let perm = PERMS[3], reversed = 1, driver = 'learned', word = 'HELLO';
  let agent = null, rival = null, sim = new Sim('write', word, 'motors'), shown = sim.theta.slice(), moves = 0, acc = 0, lastFrame = 0, holdUntil = 0, lastResult = '', pending = '';

  function buildArm() {
    const arm = factoryArm('motors');
    if (on.swap) arm.perm = perm.slice();
    if (on.reverse) arm.gain[reversed] *= -1;
    arm.gain = arm.gain.map(g => g * strength);
    arm.crosstalk = drag;
    return arm;
  }
  function restart() {
    sim = new Sim('write', word, 'motors'); sim.arm = buildArm(); sim.reset();
    if (agent) agent.reset();
    if (rival) rival.reset();
    shown = sim.theta.slice(); moves = 0; acc = 0; holdUntil = 0;
    if (pending) { lastResult = pending; pending = ''; }
    renderStatus();
  }
  const inkable = () => sim.path.filter(p => p.down).length;
  const inked = () => sim.path.reduce((n, p, i) => n + (p.down && sim.touched[i] ? 1 : 0), 0);
  const NAMES = { learned: 'Our meta-RL agent', recurrent: 'Recurrent agent', calibrate: 'Hand-written self-calibrating controller', factory: 'Controller that cannot adapt' };
  const who = () => NAMES[driver];
  const brain = () => driver === 'learned' ? agent : driver === 'recurrent' ? rival : null;
  const handWritten = () => driver === 'factory' || driver === 'calibrate';
  function renderStatus() {
    $('status').textContent = !handWritten() && !brain() ? 'Loading the trained agent.' : `${who()}: ${inked()} of ${inkable()} points inked, ${moves} moves`;
    $('last').textContent = lastResult;
  }
  function finish() {
    pending = `Last word: ${who().toLowerCase()} inked ${inked()} of ${inkable()} points in ${moves} moves.`;
    holdUntil = performance.now() + HOLD_MS; renderStatus();
    $('status').textContent = `${who()}: finished with ${inked()} of ${inkable()} points inked in ${moves} moves`;
  }
  function step() {
    const b = brain(); let a;
    if (b) {
      if (moves && moves % EPISODE_STEPS === 0) b.reset();              // both agents were trained on 150-move episodes
      a = b.sample(b.act(sim.obs(), Math.min(sim.trialT, TRIAL_STEPS), TRIAL_STEPS).logits, TEMPERATURE);
    } else a = sim.choose(driver);
    sim.step(a); moves++;
    if (b) b.remember(a, sim.lastStep.obs, sim.lastStep.settled, sim.lastStep.reward);
    if (sim.done || moves >= MAX_MOVES) finish();
  }

  // ---- controls ----
  $('damage').innerHTML = DAMAGE.map(([id, label]) =>
    `<button type="button" class="chip" id="dmg-${id}" aria-pressed="false">${label}</button>`).join('') +
    `<button type="button" class="chip fix" id="fix">Fix everything</button>`;
  for (const [id] of DAMAGE) $('dmg-' + id).addEventListener('click', e => {
    on[id] = !on[id]; e.currentTarget.setAttribute('aria-pressed', on[id]);
    if (id === 'swap' && on.swap) perm = PERMS[Math.floor(Math.random() * PERMS.length)];
    if (id === 'reverse' && on.reverse) reversed = Math.floor(Math.random() * N);
    restart();
  });
  // The sliders: the label follows the thumb, the word starts again when the thumb is let go.
  function renderSliders() {
    const inRange = strength >= TRAINED_STRENGTH[0] - 1e-9 && strength <= TRAINED_STRENGTH[1] + 1e-9;
    $('strength-val').textContent = `${strength.toFixed(1)}× · ${inRange ? 'practiced' : 'never practiced'}`;
    $('strength-val').classList.toggle('out', !inRange);
    $('strength').setAttribute('aria-valuetext', $('strength-val').textContent);
    $('drag-val').textContent = drag ? `${Math.round(drag * 100)}% · never practiced` : 'none';
    $('drag-val').classList.toggle('out', drag > 0);
    $('drag').setAttribute('aria-valuetext', $('drag-val').textContent);
  }
  $('strength').addEventListener('input', e => { strength = +e.target.value; renderSliders(); });
  $('drag').addEventListener('input', e => { drag = +e.target.value; renderSliders(); });
  $('strength').addEventListener('change', restart);
  $('drag').addEventListener('change', restart);
  $('fix').addEventListener('click', () => {
    for (const [id] of DAMAGE) { on[id] = false; $('dmg-' + id).setAttribute('aria-pressed', 'false'); }
    strength = 1; drag = 0; $('strength').value = 1; $('drag').value = 0; renderSliders(); restart();
  });
  renderSliders();
  document.querySelectorAll('input[name=driver]').forEach(r => r.addEventListener('change', () => { driver = r.value; restart(); }));
  $('word').addEventListener('input', e => {
    const clean = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 5);
    if (clean !== e.target.value) e.target.value = clean;
    if (clean && clean !== word) { word = clean; restart(); }
  });

  // ---- drawing ----
  const cv = $('view'), ctx = cv.getContext('2d');
  function draw() {
    const css = getComputedStyle(document.documentElement), col = n => css.getPropertyValue(n).trim();
    const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.round(cv.clientWidth * dpr) || 900, H = Math.round(W * 2 / 3);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const L = sim.arm.lengths.reduce((a, b) => a + b, 0), S = W / (1.9 * L), cx = W / 2, cy = H * 0.75;
    const X = p => cx + p[0] * S, Y = p => cy - p[1] * S;
    const line = (a, b) => { ctx.beginPath(); ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)); ctx.stroke(); };
    ctx.clearRect(0, 0, W, H); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    // the word as it should look
    ctx.strokeStyle = col('--line'); ctx.lineWidth = 0.05 * S;
    for (const s of sim.strokes) for (let i = 1; i < s.length; i++) line(s[i - 1], s[i]);
    // the pieces whose points the pen has touched
    ctx.strokeStyle = col('--accent'); ctx.lineWidth = 0.05 * S;
    for (let i = 1; i < sim.path.length; i++) if (sim.path[i].down && sim.touched[i]) line(sim.path[i - 1].p, sim.path[i].p);
    // where the pen really went, most recent moves only
    const tr = sim.trace, from = Math.max(1, tr.length - 60);
    ctx.strokeStyle = col('--ink2'); ctx.lineWidth = 1.2 * dpr;
    for (let i = from; i < tr.length; i++) { ctx.globalAlpha = 0.1 + 0.5 * (i - from) / 60; line(tr[i - 1], tr[i]); }
    ctx.globalAlpha = 1;
    if (!holdUntil && !sim.done) {                       // the point the pen is after
      ctx.strokeStyle = col('--target'); ctx.lineWidth = 2 * dpr;
      ctx.beginPath(); ctx.arc(X(sim.target), Y(sim.target), sim.reach * S, 0, 7); ctx.stroke();
    }
    const pts = [[0, 0]]; let c = 0;
    for (let i = 0; i < N; i++) { c += shown[i]; const q = pts[i]; pts.push([q[0] + sim.arm.lengths[i] * Math.cos(c), q[1] + sim.arm.lengths[i] * Math.sin(c)]); }
    ctx.strokeStyle = col('--arm'); ctx.globalAlpha = 0.9;
    for (let i = 0; i < N; i++) { ctx.lineWidth = (10 - 2.4 * i) * dpr; line(pts[i], pts[i + 1]); }
    ctx.globalAlpha = 1;
    for (let j = 0; j < N; j++) {
      ctx.fillStyle = col('--bench'); ctx.strokeStyle = col('--arm'); ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath(); ctx.arc(X(pts[j]), Y(pts[j]), 6 * dpr, 0, 7); ctx.fill(); ctx.stroke();
    }
    ctx.fillStyle = col('--accent'); ctx.beginPath(); ctx.arc(X(pts[N]), Y(pts[N]), 5 * dpr, 0, 7); ctx.fill();
  }

  function frame(now) {
    const dt = Math.min(0.1, (now - lastFrame) / 1000 || 0); lastFrame = now;
    if (holdUntil) { if (now >= holdUntil) restart(); }
    else if (handWritten() || brain()) {
      acc = Math.min(acc + dt * MOVES_PER_SECOND, 8);
      const budget = performance.now() + 14;
      while (acc >= 1 && !holdUntil && performance.now() < budget) { acc -= 1; step(); }
      if (!holdUntil) renderStatus();
    }
    for (let i = 0; i < N; i++) shown[i] = reduce ? sim.theta[i] : shown[i] + (sim.theta[i] - shown[i]) * Math.min(1, dt * 22);
    draw();
    requestAnimationFrame(frame);
  }

  function boot() {
    const pack = (window.ARM_AGENTS || {}).motors3;
    if (pack) agent = new window.ArmAgent(pack);
    const other = (window.ARM_RIVALS || {}).recurrent3;
    if (other) rival = new window.RecurrentAgent(other);
    restart(); requestAnimationFrame(frame);
  }
  boot();
})();
