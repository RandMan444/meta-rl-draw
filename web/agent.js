/* The trained agent's forward pass in plain JavaScript, one arm at a time.
 *
 * It mirrors A2C_EPN.forward in python/arm_meta.py block by block:
 *   1. a frozen "situation key" for every candidate action
 *   2. ONE hard nearest neighbor per action from this episode's memory (cosine on the frozen keys),
 *      whose observed effect is added to the action's latent:  z_hat = z + delta
 *   3. a one-layer transformer over the memory rows, mean pooled
 *   4. a per-action actor and a critic
 * check_agent.js replays a PyTorch trace through this file and compares the logits.
 */
(function (root) {
  'use strict';

  function decode(t) {
    const bin = typeof atob === 'function' ? atob(t.b64) : Buffer.from(t.b64, 'base64').toString('binary');
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function decodeHalf(t) {               // float16, used for the training snapshots to keep them small
    const bin = typeof atob === 'function' ? atob(t.h16) : Buffer.from(t.h16, 'base64').toString('binary');
    const out = new Float32Array(bin.length / 2);
    for (let i = 0; i < out.length; i++) {
      const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8), e = (h >> 10) & 31, f = h & 1023;
      const v = e === 0 ? f / 1024 * 2 ** -14 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15);
      out[i] = h & 0x8000 ? -v : v;
    }
    return out;
  }
  const elu = v => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = Math.exp(v[i]) - 1; return v; };
  const relu = v => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; return v; };
  // y = W x + b for `rows` stacked inputs. W is [nOut, nIn] row-major, as PyTorch stores it.
  function linear(W, b, x, nIn, nOut, rows) {
    const y = new Float32Array(rows * nOut);
    for (let r = 0; r < rows; r++) {
      const xo = r * nIn, yo = r * nOut;
      for (let o = 0; o < nOut; o++) {
        let s = b ? b[o] : 0; const wo = o * nIn;
        for (let i = 0; i < nIn; i++) s += W[wo + i] * x[xo + i];
        y[yo + o] = s;
      }
    }
    return y;
  }
  function layerNorm(x, g, b, d, rows) {
    for (let r = 0; r < rows; r++) {
      let mean = 0, v = 0; const o = r * d;
      for (let i = 0; i < d; i++) mean += x[o + i];
      mean /= d;
      for (let i = 0; i < d; i++) { const c = x[o + i] - mean; v += c * c; }
      const inv = 1 / Math.sqrt(v / d + 1e-5);
      for (let i = 0; i < d; i++) x[o + i] = (x[o + i] - mean) * inv * g[i] + b[i];
    }
    return x;
  }

  class ArmAgent {
    constructor(pack) {
      this.meta = pack;
      this.A = pack.actions; this.OBS = pack.obs; this.M = pack.memory;
      this.sliceIdx = pack.slice_idx; this.K = pack.slice_idx[0].length;
      const w = {}; for (const k in pack.tensors) w[k] = decode(pack.tensors[k]);
      this.w = w;
      this.P = pack.tensors.protos.shape[0]; this.KEY = pack.tensors.protos.shape[1];
      this.gate = w['keynet.gate_logits'].map(v => 1 / (1 + Math.exp(-v)));
      this.protoNorm = new Float32Array(this.P);
      for (let p = 0; p < this.P; p++) { let s = 0; for (let i = 0; i < this.KEY; i++) s += w.protos[p * this.KEY + i] ** 2; this.protoNorm[p] = s; }
      this.D = 66; this.Z = 8; this.HEADS = 3; this.rowDim = 2 * this.K + 4;
      this.reset();
    }
    /* Swap in the weights training had reached at an earlier episode (see python/export_replay.py).
       The frozen keys stay; passing nothing restores the fully trained agent. */
    useSnapshot(snap) {
      if (!this.trained) this.trained = Object.assign({}, this.w);
      if (!snap) { this.w = Object.assign({}, this.trained); return; }
      if (!snap.decoded) { snap.decoded = {}; for (const k in snap.tensors) snap.decoded[k] = decodeHalf(snap.tensors[k]); }
      this.w = Object.assign({}, this.trained, snap.decoded);
    }
    reset() {                       // a new episode: the memory is emptied
      this.n = 0;
      this.rows = new Float32Array(this.M * this.rowDim);
      this.memKeys = new Float32Array(this.M * this.KEY);
    }
    slice(obs, a) {                 // the features this action's gate kept
      const x = new Float32Array(this.K), on = this.w.slice_on;
      for (let i = 0; i < this.K; i++) x[i] = obs[this.sliceIdx[a][i]] * on[a * this.K + i];
      return x;
    }
    keys(obs) {                     // one unit-length key per candidate action
      const { A, OBS, w } = this, nIn = OBS + A, x = new Float32Array(A * nIn);
      for (let a = 0; a < A; a++) {
        for (let i = 0; i < OBS; i++) x[a * nIn + i] = obs[i] * this.gate[a * OBS + i];
        x[a * nIn + OBS + a] = 1;
      }
      let h = elu(linear(w['keynet.net.0.weight'], w['keynet.net.0.bias'], x, nIn, 256, A));
      h = elu(linear(w['keynet.net.2.weight'], w['keynet.net.2.bias'], h, 256, 256, A));
      const k = linear(w['keynet.net.4.weight'], w['keynet.net.4.bias'], h, 256, this.KEY, A);
      for (let a = 0; a < A; a++) {
        let s = 0; for (let i = 0; i < this.KEY; i++) s += k[a * this.KEY + i] ** 2;
        const inv = 1 / Math.max(Math.sqrt(s), 1e-12);
        for (let i = 0; i < this.KEY; i++) k[a * this.KEY + i] *= inv;
      }
      return k;
    }
    codes(keys) {                   // nearest prototype for each key
      const { A, P, KEY, w } = this, out = new Int32Array(A);
      for (let a = 0; a < A; a++) {
        let best = Infinity, bi = 0;
        for (let p = 0; p < P; p++) {
          let dot = 0; const po = p * KEY, ko = a * KEY;
          for (let i = 0; i < KEY; i++) dot += w.protos[po + i] * keys[ko + i];
          const d = this.protoNorm[p] - 2 * dot;
          if (d < best) { best = d; bi = p; }
        }
        out[a] = bi;
      }
      return out;
    }
    transformer(zb, za) {           // mean-pooled transformer encoder layer over the filled memory rows
      const { n, D, K, Z, M, rowDim, w, HEADS } = this, inDim = rowDim - 1 + 2 * Z + 1;
      const summary = new Float32Array(D);
      if (n === 0) return summary;
      const x0 = new Float32Array(n * inDim);
      for (let j = 0; j < n; j++) {
        const o = j * inDim;
        for (let i = 1; i < rowDim; i++) x0[o + i - 1] = this.rows[j * rowDim + i];
        for (let i = 0; i < Z; i++) { x0[o + rowDim - 1 + i] = zb[j * Z + i]; x0[o + rowDim - 1 + Z + i] = za[j * Z + i]; }
        x0[o + inDim - 1] = j / M;
      }
      let x = elu(linear(w['transformer.row_encoder.0.weight'], w['transformer.row_encoder.0.bias'], x0, inDim, D, n));
      x = linear(w['transformer.row_encoder.2.weight'], w['transformer.row_encoder.2.bias'], x, D, D, n);
      const P_ = 'transformer.attention.', qkv = linear(w[P_ + 'self_attn.in_proj_weight'], w[P_ + 'self_attn.in_proj_bias'], x, D, 3 * D, n);
      const hd = D / HEADS, scale = 1 / Math.sqrt(hd), mixed = new Float32Array(n * D), att = new Float32Array(n);
      for (let h = 0; h < HEADS; h++) {
        for (let i = 0; i < n; i++) {
          let mx = -Infinity;
          for (let j = 0; j < n; j++) {
            let s = 0; const qo = i * 3 * D + h * hd, ko = j * 3 * D + D + h * hd;
            for (let c = 0; c < hd; c++) s += qkv[qo + c] * qkv[ko + c];
            att[j] = s * scale; if (att[j] > mx) mx = att[j];
          }
          let sum = 0; for (let j = 0; j < n; j++) { att[j] = Math.exp(att[j] - mx); sum += att[j]; }
          for (let j = 0; j < n; j++) {
            const p = att[j] / sum, vo = j * 3 * D + 2 * D + h * hd, oo = i * D + h * hd;
            for (let c = 0; c < hd; c++) mixed[oo + c] += p * qkv[vo + c];
          }
        }
      }
      const sa = linear(w[P_ + 'self_attn.out_proj.weight'], w[P_ + 'self_attn.out_proj.bias'], mixed, D, D, n);
      for (let i = 0; i < x.length; i++) x[i] += sa[i];
      layerNorm(x, w[P_ + 'norm1.weight'], w[P_ + 'norm1.bias'], D, n);
      const ff = linear(w[P_ + 'linear2.weight'], w[P_ + 'linear2.bias'],
        relu(linear(w[P_ + 'linear1.weight'], w[P_ + 'linear1.bias'], x, D, 2 * D, n)), 2 * D, D, n);
      for (let i = 0; i < x.length; i++) x[i] += ff[i];
      layerNorm(x, w[P_ + 'norm2.weight'], w[P_ + 'norm2.bias'], D, n);
      for (let j = 0; j < n; j++) for (let i = 0; i < D; i++) summary[i] += x[j * D + i] / n;
      return summary;
    }
    /* obs: the arm's observation; trialT: steps already spent on this target.
       Returns the logits, the value, and what `remember` needs after the step. */
    act(obs, trialT, trialSteps) {
      const { A, K, Z, D, n, w, rowDim, KEY } = this;
      const keys = this.keys(obs), codes = this.codes(keys);
      // memory rows: [action, before slice, after slice, reward, useless, time in trial]
      const before = new Float32Array(n * K), after = new Float32Array(n * K);
      for (let j = 0; j < n; j++) for (let i = 0; i < K; i++) { before[j * K + i] = this.rows[j * rowDim + 1 + i]; after[j * K + i] = this.rows[j * rowDim + 1 + K + i]; }
      const zb = linear(w['phi.weight'], w['phi.bias'], before, K, Z, n), za = linear(w['phi.weight'], w['phi.bias'], after, K, Z, n);
      const summary = this.transformer(zb, za);
      const o1 = new Float32Array(this.OBS + 1); o1.set(obs); o1[this.OBS] = trialT / trialSteps;
      const enc = linear(w['encoder.2.weight'], w['encoder.2.bias'], elu(linear(w['encoder.0.weight'], w['encoder.0.bias'], o1, this.OBS + 1, D, 1)), D, D, 1);
      const ctx = new Float32Array(2 * D); ctx.set(enc); ctx.set(summary, D);
      const value = linear(w['critic.2.weight'], w['critic.2.bias'], elu(linear(w['critic.0.weight'], w['critic.0.bias'], ctx, 2 * D, 256, 1)), 256, 1, 1)[0];
      const fDim = K + 3 * Z + 16 + 2 * D, feats = new Float32Array(A * fDim), slices = [], nearest = new Int32Array(A).fill(-1);
      for (let a = 0; a < A; a++) {
        const x = this.slice(obs, a); slices.push(x);
        const z = linear(w['phi.weight'], w['phi.bias'], x, K, Z, 1);
        // the single hard nearest neighbor: the most similar stored key whose row was not "useless"
        let best = -Infinity, bj = -1;
        for (let j = 0; j < n; j++) {
          if (this.rows[j * rowDim + 2 * K + 2] >= 0.5) continue;
          let s = 0; for (let i = 0; i < KEY; i++) s += keys[a * KEY + i] * this.memKeys[j * KEY + i];
          if (s > best) { best = s; bj = j; }
        }
        nearest[a] = bj;
        const o = a * fDim;
        feats.set(x, o); feats.set(z, o + K);
        for (let i = 0; i < Z; i++) {
          const delta = bj < 0 ? 0 : za[bj * Z + i] - zb[bj * Z + i];
          feats[o + K + Z + i] = z[i] + delta; feats[o + K + 2 * Z + i] = delta;
        }
        for (let i = 0; i < 16; i++) feats[o + K + 3 * Z + i] = w['code_embedding.weight'][codes[a] * 16 + i];
        feats.set(ctx, o + K + 3 * Z + 16);
      }
      const logits = linear(w['actor.2.weight'], w['actor.2.bias'], elu(linear(w['actor.0.weight'], w['actor.0.bias'], feats, fDim, 256, A)), 256, 1, A);
      this._last = { keys, slices, trialT, trialSteps };
      return { logits, value, codes, nearest };
    }
    /* temperature 1 samples the way training did; lower is more decisive; 0 always takes the top action. */
    sample(logits, temperature = 1) {
      let mx = -Infinity, arg = 0; for (let a = 0; a < logits.length; a++) if (logits[a] > mx) { mx = logits[a]; arg = a; }
      if (!temperature) return arg;
      const p = Array.from(logits, v => Math.exp((v - mx) / temperature)), total = p.reduce((s, v) => s + v, 0);
      let r = Math.random() * total;
      for (let a = 0; a < p.length; a++) { r -= p[a]; if (r <= 0) return a; }
      return arg;
    }
    /* Store what the chosen action did. settledObs is the observation after the move with the SAME target. */
    remember(action, obs, settledObs, reward) {
      if (this.n >= this.M) return;
      const { K, rowDim, KEY } = this, o = this.n * rowDim, last = this._last;
      const moved = Math.hypot(settledObs[0] - obs[0], settledObs[1] - obs[1]);
      this.rows[o] = action;
      this.rows.set(last.slices[action], o + 1);
      this.rows.set(this.slice(settledObs, action), o + 1 + K);
      this.rows[o + 2 * K + 1] = reward;
      this.rows[o + 2 * K + 2] = moved < 0.004 ? 1 : 0;       // "useless": the pen did not move
      this.rows[o + 2 * K + 3] = last.trialT / last.trialSteps;
      for (let i = 0; i < KEY; i++) this.memKeys[this.n * KEY + i] = last.keys[action * KEY + i];
      this.n++;
    }
  }

  root.ArmAgent = ArmAgent;
  if (typeof module !== 'undefined') module.exports = { ArmAgent };
})(typeof window !== 'undefined' ? window : globalThis);
