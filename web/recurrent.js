/* The recurrent baseline (python/arm_rl2.py) in plain JavaScript: an LSTM that is fed the observation, its previous
   action and the previous reward, and has to keep everything it learns about the arm in its hidden state. */
(function (root) {
  function decodeHalf(b64) {
    const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const out = new Float32Array(bin.length / 2);
    for (let i = 0; i < out.length; i++) {
      const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8), e = (h >> 10) & 31, f = h & 1023;
      const v = e === 0 ? f / 1024 * 2 ** -14 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15);
      out[i] = h & 0x8000 ? -v : v;
    }
    return out;
  }
  // y = W x + b, W is [nOut, nIn] row-major as PyTorch stores it
  function linear(W, b, x, nOut, y) {
    const nIn = x.length;
    for (let o = 0; o < nOut; o++) { let s = b[o]; const base = o * nIn; for (let i = 0; i < nIn; i++) s += W[base + i] * x[i]; y[o] += s; }
    return y;
  }
  const sigmoid = v => 1 / (1 + Math.exp(-v));

  class RecurrentAgent {
    constructor(pack) {
      this.A = pack.actions; this.OBS = pack.obs; this.H = pack.hidden; this.trainedEpisodes = pack.trained_episodes;
      this.w = {}; for (const k in pack.tensors) this.w[k] = decodeHalf(pack.tensors[k].h16);
      this.E = pack.tensors['encoder.0.weight'].shape[0];
      this.reset();
    }
    reset() {                       // a new episode: the hidden state is cleared
      this.h = new Float32Array(this.H); this.c = new Float32Array(this.H); this.prevAction = -1; this.prevReward = 0;
    }
    act(obs, trialT, trialSteps) {
      const { A, OBS, H, E, w } = this, x = new Float32Array(OBS + 1 + A + 1);
      x.set(obs); x[OBS] = trialT / trialSteps;
      if (this.prevAction >= 0) x[OBS + 1 + this.prevAction] = 1;
      x[OBS + 1 + A] = this.prevReward;
      const e = linear(w['encoder.0.weight'], w['encoder.0.bias'], x, E, new Float32Array(E));
      for (let i = 0; i < E; i++) if (e[i] < 0) e[i] = Math.exp(e[i]) - 1;
      const g = new Float32Array(4 * H);                      // gates in PyTorch's order: input, forget, cell, output
      linear(w['lstm.weight_ih'], w['lstm.bias_ih'], e, 4 * H, g); linear(w['lstm.weight_hh'], w['lstm.bias_hh'], this.h, 4 * H, g);
      for (let i = 0; i < H; i++) {
        this.c[i] = sigmoid(g[H + i]) * this.c[i] + sigmoid(g[i]) * Math.tanh(g[2 * H + i]);
        this.h[i] = sigmoid(g[3 * H + i]) * Math.tanh(this.c[i]);
      }
      return { logits: linear(w['actor.weight'], w['actor.bias'], this.h, A, new Float32Array(A)) };
    }
    sample(logits, temperature = 1) {
      let mx = -Infinity, arg = 0; for (let a = 0; a < logits.length; a++) if (logits[a] > mx) { mx = logits[a]; arg = a; }
      if (!temperature) return arg;
      const p = Array.from(logits, v => Math.exp((v - mx) / temperature)), total = p.reduce((s, v) => s + v, 0);
      let r = Math.random() * total;
      for (let a = 0; a < p.length; a++) { r -= p[a]; if (r <= 0) return a; }
      return arg;
    }
    remember(action, obs, settledObs, reward) { this.prevAction = action; this.prevReward = reward; }
  }

  root.RecurrentAgent = RecurrentAgent;
  if (typeof module !== 'undefined') module.exports = { RecurrentAgent };
})(typeof window !== 'undefined' ? window : globalThis);
