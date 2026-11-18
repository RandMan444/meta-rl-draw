"""A meta-RL agent with nearest-neighbor retrieval, on the broken-arm environment.

The algorithm was first developed on DeepMind's Alchemy benchmark and is ported here block by
block: random warm-up, a SituationKey with per-action gates pretrained with the surprise_bits
loss, radius-cover codes, learned action slices, an episodic memory with a frozen key per row,
ONE literal nearest neighbor per candidate action (argmax cosine of frozen keys),
z_hat = z + delta, a transformer over the memory rows, a per-action actor, a critic, and plain
n-step A2C. The hyperparameters are the ones used on Alchemy. Only the environment-specific
pieces changed:

  * effect id = quantised pen displacement (direction x size) instead of a stone-feature hash
  * a "useless" memory row = the pen did not move (dead motor or joint limit)
  * memory rows store the observation after the move but before the target changes, so no
    row has to be dropped at a trial boundary

  python arm_meta.py <out_dir> [--no-nn] [--episodes 1000] [--key-steps 16000]
"""
import argparse, csv, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from arm_env import ArmEnv, N_ACTIONS, OBS_DIM, TRIAL_STEPS, EPISODE_STEPS

config = {
    'n-actions': N_ACTIONS, 'obs-dim': OBS_DIM, 'dict-len': 150, 'n-workers': 16,
    'n-episodes': 1000, 'n-key-steps': 16000, 'n-warmup-episodes': 300,
    'lr': 7.5e-4, 'gamma': 0.7, 'value-loss-weight': 0.5, 'entropy-weight': 0.05,
    'grad-clip-norm': 100, 'n-step-update': 20, 'device': 'cpu',
}
SEED_WARMUP, SEED_TRAIN, SEED_EVAL = 10_000, 20_000, 30_000


# ---------- warmup: random-policy experience, only to pretrain the situation keys ----------
def collect_random(env, n_episodes):
    rng = np.random.default_rng(0)
    states, actions, next_states, rewards, episode_ids = [], [], [], [], []
    ep_base = 0
    for _ in range(math.ceil(n_episodes / env.n)):
        s = env.reset()
        done = np.zeros(env.n, bool)
        while not done.all():
            a = rng.integers(0, N_ACTIONS, size=env.n)
            s1, r, done, new_trial, settled, _ = env.step(a)
            states.append(s); actions.append(a); next_states.append(settled); rewards.append(r)
            episode_ids.append(ep_base + np.arange(env.n))
            s = s1
        ep_base += env.n
    return (np.concatenate(states).astype(np.float32), np.concatenate(actions),
            np.concatenate(next_states).astype(np.float32), np.concatenate(rewards).astype(np.float32),
            np.concatenate(episode_ids))


def effect_id(s, s1):
    """What the action did to the pen: 0 = nothing, else direction (12) x size (3)."""
    d = np.asarray(s1)[:, :2] - np.asarray(s)[:, :2]
    mag = np.linalg.norm(d, axis=-1)
    direction = ((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * 12).astype(np.int64).clip(max=11)
    size = (mag > 0.03).astype(np.int64) + (mag > 0.08).astype(np.int64)
    return np.where(mag < 0.004, 0, 1 + direction * 3 + size)


# ---------- pretrained situation key; a learned per-action gate filters irrelevant features ----------
class SituationKey(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=256, z_dim=32):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(nn.Linear(obs_dim + n_actions, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, z_dim))
        self.gate_logits = nn.Parameter(torch.full((n_actions, obs_dim), 2.0))
        self.protos = torch.zeros(1, z_dim)

    def gate(self, a):
        return torch.sigmoid(self.gate_logits[a])

    def embed(self, s, a):
        x = torch.cat([s * self.gate(a), F.one_hot(a, self.n_actions).float()], -1)
        return F.normalize(self.net(x), dim=-1)

    @torch.no_grad()
    def code(self, s, a):
        return torch.cdist(self.embed(s, a), self.protos.to(s.device)).argmin(-1)


def surprise_bits(weights, effect, episode, prior, cross_episode=False, alpha=0.01):
    same_ep = (episode[:, None] == episode[None, :]).float()
    same_ep.fill_diagonal_(0)
    same_effect = (effect[:, None] == effect[None, :]).float()
    pos = same_effect * same_ep
    if cross_episode:
        valid = (effect != 0).float()
        pos = torch.maximum(pos, same_effect * valid[:, None] * valid[None, :] *
                            (1 - torch.eye(len(effect), device=effect.device)))
    neg = (1 - same_effect) * same_ep
    p = ((weights * pos).sum(1) + alpha * prior) / ((weights * pos).sum(1) + (weights * neg).sum(1) + alpha)
    return -torch.log2(p)


def key_loss(z, effect, episode, prior):
    weights = torch.exp((z @ z.T) / 0.1)
    main = surprise_bits(weights, effect, episode, prior).mean()
    return main + 0.3 * surprise_bits(weights, effect, episode, prior, cross_episode=True).mean()


def code_metrics(codes, effects, episodes, priors):
    bits, conflicts, pairs = [], 0, 0
    for episode in np.unique(episodes):
        idx = np.where(episodes == episode)[0]
        c = torch.as_tensor(codes[idx]); f = torch.as_tensor(effects[idx])
        matches = (c[:, None] == c[None, :]).float()
        bits.append(surprise_bits(matches, f, torch.zeros(len(idx)), torch.as_tensor(priors[idx]).float(), alpha=1.0))
        same = (matches > 0) & ~torch.eye(len(idx), dtype=torch.bool)
        pairs += int(same.sum())
        conflicts += int((same & (f[:, None] != f[None, :])).sum())
    return float(torch.cat(bits).mean()), conflicts / max(1, pairs)


def radius_cover(z, radius, max_protos=4000):
    points = torch.as_tensor(z)
    protos = torch.zeros(0, points.shape[1])
    for item in points:
        if len(protos) == 0 or (torch.cdist(item[None], protos) > radius).all():
            protos = torch.cat([protos, item[None]])
            if len(protos) >= max_protos:
                break
    return protos.numpy()


def pretrain_situation_key(S, A, S1, EP, steps, log):
    effects = effect_id(S, S1)
    ep_ids = np.unique(EP)
    rng = np.random.default_rng(1)
    rng.shuffle(ep_ids)
    val_eps = set(ep_ids[:max(1, len(ep_ids) // 5)].tolist())
    is_val = np.isin(EP, list(val_eps))
    train_eps = np.array([e for e in ep_ids if e not in val_eps])
    unique, inv, counts = np.unique(effects, return_inverse=True, return_counts=True)
    prior = (counts / counts.sum())[inv].astype(np.float32)
    S_t, A_t = torch.as_tensor(S), torch.as_tensor(A)
    E_t, EP_t, PR_t = torch.as_tensor(effects), torch.as_tensor(EP), torch.as_tensor(prior)
    by_ep = {e: np.where(EP == e)[0] for e in ep_ids}
    keynet = SituationKey()
    optimizer = optim.AdamW(keynet.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, steps + 1):
        episodes = rng.choice(train_eps, min(8, len(train_eps)), replace=False)
        idx = torch.as_tensor(np.concatenate([by_ep[e] for e in episodes]))
        loss = key_loss(keynet.embed(S_t[idx], A_t[idx]), E_t[idx], EP_t[idx], PR_t[idx])
        loss = loss + 4.0 * keynet.gate(A_t[idx]).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 1000 == 0:
            log(f'key pretrain {step}: loss={loss.item():.3f}')
    keynet.eval()
    with torch.no_grad():
        Z = keynet.embed(S_t, A_t).numpy()
    sub = np.random.default_rng(2).choice(np.where(~is_val)[0], min(20000, int((~is_val).sum())), replace=False)
    vsub = np.where(is_val)[0]
    best = None
    for radius in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        P = radius_cover(Z[sub], radius)
        codes_val = torch.cdist(torch.as_tensor(Z[vsub]), torch.as_tensor(P)).argmin(-1).numpy()
        bits, conflicts = code_metrics(codes_val, effects[vsub], EP[vsub], prior[vsub])
        log(f'  radius {radius}: {len(P)} codes, {bits:.3f} bits, conflicts {conflicts:.3f}')
        if conflicts <= 0.02 and (best is None or bits < best[0]):
            best = (bits, radius, P)
    if best is None:                       # effects are continuous here, so pick the fewest bits instead
        cands = []
        for radius in [0.2, 0.3, 0.4, 0.5]:
            P = radius_cover(Z[sub], radius)
            codes_val = torch.cdist(torch.as_tensor(Z[vsub]), torch.as_tensor(P)).argmin(-1).numpy()
            cands.append((code_metrics(codes_val, effects[vsub], EP[vsub], prior[vsub])[0], radius, P))
        best = min(cands, key=lambda c: c[0])
    keynet.protos = torch.as_tensor(best[2], dtype=torch.float32)
    log(f'key radius={best[1]}, codes={len(keynet.protos)}')
    return keynet


def build_action_slices(keynet):
    with torch.no_grad():
        gate = torch.sigmoid(keynet.gate_logits)
        on = gate >= 0.1 * gate.max(dim=1, keepdim=True).values
    sizes = on.sum(1)
    focused = sizes[sizes <= OBS_DIM // 2]
    width = int(focused.max()) if len(focused) else min(int(sizes.max()), 12)
    idx = np.zeros((N_ACTIONS, width), np.int64)
    active = np.zeros((N_ACTIONS, width), np.float32)
    for a in range(N_ACTIONS):
        selected = np.where(on[a].numpy())[0][:width]
        idx[a, :len(selected)] = selected
        active[a, :len(selected)] = 1
    return idx, active


def make_row_fn(slice_idx, slice_on):
    def slices(s):
        return np.asarray(s, np.float32)[:, slice_idx] * slice_on[None]

    def make_row(actions, s, s1, r, t_in_trial):
        B = np.arange(len(actions))
        before, after = slices(s)[B, actions], slices(s1)[B, actions]
        bad = (effect_id(s, s1) == 0).astype(np.float32)            # "useless": the pen did not move
        return np.concatenate([actions[:, None].astype(np.float32), before, after, r[:, None], bad[:, None],
                               np.asarray(t_in_trial, np.float32)[:, None] / TRIAL_STEPS], -1).astype(np.float32)
    return make_row


# ---------- episodic memory with a frozen key per row ----------
class Episodic_Memory:
    def __init__(self, n_workers, mem_size, mem_dim, key_dim=32):
        self.mem_dim, self.mem_size, self.n_workers, self.key_dim = mem_dim, mem_size, n_workers, key_dim
        self.reset()

    def reset(self):
        self.pointer = 0
        self.memory = np.zeros((self.n_workers, self.mem_size, self.mem_dim), np.float32)
        self.key_memory = np.zeros((self.n_workers, self.mem_size, self.key_dim), np.float32)
        self.key_present = np.zeros((self.n_workers, self.mem_size), np.float32)

    def push(self, memories, keys):
        if self.pointer < self.mem_size:
            self.memory[:, self.pointer, :] = memories
            self.key_memory[:, self.pointer, :] = keys
            self.key_present[:, self.pointer] = 1.0
            self.pointer += 1

    def generate_mask(self):
        mask = np.zeros((self.n_workers, self.mem_size))
        mask[:, :self.pointer] = 1
        return mask


class EPN_TRANSFORMER(nn.Module):
    def __init__(self, mem_dim, z_dim=8, d=66, n_heads=3):
        super().__init__()
        self.row_encoder = nn.Sequential(nn.Linear(mem_dim - 1 + 2 * z_dim + 1, d), nn.ELU(), nn.Linear(d, d))
        self.attention = nn.TransformerEncoderLayer(d, n_heads, d * 2, dropout=0.0, batch_first=True)
        self.full_size = None      # memory capacity, so a row's age means the same when only the filled rows are passed

    def forward(self, mem, before_latent, after_latent, mask):
        B, M = mem.shape[:2]
        age = torch.arange(M, device=mem.device).float()[None, :, None].expand(B, M, 1) / (self.full_size or M)
        rows = self.row_encoder(torch.cat([mem[..., 1:], before_latent, after_latent, age], -1))
        pad_mask = torch.where((mask.sum(1) > 0)[:, None], mask == 0, torch.zeros_like(mask, dtype=torch.bool))
        attended = self.attention(rows, src_key_padding_mask=pad_mask)
        return (attended * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


# The only retrieval rule is j* = argmax_j <k(s,a), k(s_j,a_j)>.
class A2C_EPN(nn.Module):
    def __init__(self, keynet, slice_idx, slice_on, mem_size=150, hidden_dim=256, d=66, z_dim=8, use_nn=True):
        super().__init__()
        self.n_actions, self.mem_size, self.use_nn = N_ACTIONS, mem_size, use_nn
        self.slice_width = slice_idx.shape[1]
        self.mem_dim = 1 + 2 * self.slice_width + 3
        self.keynet = keynet
        for p in self.keynet.parameters():
            p.requires_grad_(False)
        self.key_dim = int(self.keynet.protos.shape[-1])
        self.register_buffer('slice_idx', torch.as_tensor(slice_idx))
        self.register_buffer('slice_on', torch.as_tensor(slice_on))
        K = self.slice_width
        self.phi = nn.Linear(K, z_dim)
        self.transformer = EPN_TRANSFORMER(self.mem_dim, z_dim, d, 3)
        self.transformer.full_size = mem_size
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM + 1, d), nn.ELU(), nn.Linear(d, d))
        self.code_embedding = nn.Embedding(len(keynet.protos), 16)
        self.actor = nn.Sequential(nn.Linear(K + 3 * z_dim + 16 + d + d, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(nn.Linear(d + d, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))

    @torch.no_grad()
    def codes_and_keys(self, s):
        states = torch.as_tensor(np.asarray(s, np.float32))
        B = states.shape[0]
        a = torch.arange(self.n_actions).repeat(B)
        keys = self.keynet.embed(states.repeat_interleave(self.n_actions, 0), a)
        codes = torch.cdist(keys, self.keynet.protos).argmin(-1)
        return codes.reshape(B, self.n_actions).numpy(), keys.reshape(B, self.n_actions, self.key_dim).numpy()

    def forward(self, s_t, codes_all, m_t, mask_t, query_keys, memory_keys, key_present):
        s, trial_step = s_t[:, :-1], s_t[:, -1:]
        B, K = s.shape[0], self.slice_width
        # Empty rows are masked out everywhere below, so only the filled rows are passed on. Same numbers, far less work.
        filled = max(1, int(mask_t.sum(1).max()))
        m_t, mask_t, memory_keys, key_present = m_t[:, :filled], mask_t[:, :filled], memory_keys[:, :filled], key_present[:, :filled]
        x = s[:, self.slice_idx] * self.slice_on
        z = self.phi(x)
        before = m_t[..., 1:1 + K]
        after = m_t[..., 1 + K:1 + 2 * K]
        useless_row = m_t[..., 2 * K + 2]
        zb, za = self.phi(before), self.phi(after)
        move = za - zb
        valid = (mask_t > 0) & (key_present > 0) & (useless_row < 0.5)
        similarities = torch.einsum('bad,bmd->bam', query_keys, memory_keys)
        winner = similarities.masked_fill(~valid[:, None, :], -1e9).argmax(-1)
        index_b = torch.arange(B)[:, None]
        delta = move[index_b, winner]                                  # one observed effect per action
        has_valid = valid.any(-1)
        delta = delta * has_valid[:, None, None].to(delta.dtype)
        if not self.use_nn:                                            # ablation: no retrieval
            delta = delta * 0
        z_hat = z + delta
        with torch.no_grad():
            best_similarity = similarities.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
            self._nearest_cosine = (best_similarity * has_valid[:, None]).sum() / (has_valid.sum().clamp(min=1) * self.n_actions)
        summary = self.transformer(m_t, zb, za, mask_t)
        obs = self.encoder(torch.cat([s, trial_step], -1))
        value = self.critic(torch.cat([obs, summary], -1))
        context = torch.cat([obs, summary], -1).unsqueeze(1).expand(-1, self.n_actions, -1)
        features = torch.cat([x, z, z_hat, z_hat - z, self.code_embedding(codes_all), context], -1)
        logits = self.actor(features).squeeze(-1)
        self._logits, self._value = logits.detach(), value.detach()      # read by export_web.py to check the JavaScript port
        sm = F.softmax(logits, dim=-1)
        selected_actions = Categorical(sm).sample().detach()
        lsm = F.log_softmax(logits, dim=-1)
        log_prob = lsm.gather(1, selected_actions.unsqueeze(1))
        entropy = -(lsm * sm).sum(1, keepdim=True)
        return value, selected_actions, log_prob, entropy


# ---------- plain A2C: policy, critic and entropy, no auxiliary loss ----------
def run_episode(model, optimizer, env, make_row, train=True):
    T = lambda x: torch.as_tensor(np.asarray(x)).float()
    n = env.n
    model.train(train)
    state = env.reset()
    memory = Episodic_Memory(n, model.mem_size, model.mem_dim)
    reward_sum, reach_sum = np.zeros(n, np.float32), np.zeros(n, np.float32)
    done = np.zeros(n, bool)
    cosines = []

    def act(state):
        codes_all, query_keys = model.codes_and_keys(state)
        s_in = np.concatenate([state, (env.trial_t / TRIAL_STEPS)[:, None]], 1)
        out = model(T(s_in), torch.as_tensor(codes_all), T(memory.memory), T(memory.generate_mask()),
                    T(query_keys), T(memory.key_memory), T(memory.key_present))
        return out, query_keys

    while not done.all():
        rewards_list, masks_list, values_list, log_probs, entropies = [], [], [], [], []
        for _ in range(config['n-step-update']):
            trial_t = env.trial_t.copy()
            with torch.set_grad_enabled(train):
                (value, action, log_prob, entropy), query_keys = act(state)
            cosines.append(float(model._nearest_cosine))
            actions = action.numpy()
            next_state, rewards, done, new_trial, settled, reached = env.step(actions)
            reward_sum += rewards; reach_sum += reached
            memory.push(make_row(actions, state, settled, rewards, trial_t), query_keys[np.arange(n), actions])
            state = next_state
            rewards_list.append(rewards); masks_list.append(1. - done.astype(np.float32))
            values_list.append(value); log_probs.append(log_prob); entropies.append(entropy)
            if done.all():
                break
        if not train:
            continue
        with torch.no_grad():
            G = act(state)[0][0].squeeze(-1).numpy()
        targets = []
        for r, mask in zip(rewards_list[::-1], masks_list[::-1]):
            G = r + config['gamma'] * G * mask
            targets.append(G)
        td_target = torch.as_tensor(np.stack(targets[::-1])).float().reshape(-1)
        values = torch.stack(values_list).reshape(-1)
        advantage = (td_target - values).detach()
        value_loss = F.smooth_l1_loss(values, td_target)
        action_loss = -(torch.stack(log_probs).reshape(-1) * advantage).mean()
        entropy = torch.stack(entropies).mean()
        loss = config['value-loss-weight'] * value_loss + action_loss - config['entropy-weight'] * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad-clip-norm'])
        optimizer.step()
    return float(reward_sum.mean()), float(reach_sum.mean()), float(np.mean(cosines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no-nn', action='store_true', help='ablation: same agent without nearest-neighbor retrieval')
    ap.add_argument('--episodes', type=int, default=config['n-episodes'])
    ap.add_argument('--key-steps', type=int, default=config['n-key-steps'])
    ap.add_argument('--warmup-episodes', type=int, default=config['n-warmup-episodes'])
    ap.add_argument('--keys', help='reuse a pretrained key file instead of pretraining')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--checkpoint-every', type=int, default=0, help='also save the learned weights every N episodes (for the web replay)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    logf = open(os.path.join(args.out, 'log.txt'), 'a')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    t0 = time.time()
    key_path = args.keys or os.path.join(args.out, 'keys.pt')
    if os.path.exists(key_path):
        ck = torch.load(key_path, weights_only=False)
        keynet = SituationKey(); keynet.load_state_dict(ck['net']); keynet.protos = ck['protos']; keynet.eval()
        log(f'loaded pretrained keys ({len(keynet.protos)} codes)')
    else:
        S, A, S1, R, EP = collect_random(ArmEnv(config['n-workers'], SEED_WARMUP), args.warmup_episodes)
        log(f'warmup: {len(S):,} random transitions from {len(np.unique(EP))} episodes ({time.time() - t0:.0f}s)')
        keynet = pretrain_situation_key(S, A, S1, EP, args.key_steps, log)
        torch.save({'net': keynet.state_dict(), 'protos': keynet.protos}, key_path)
        log(f'key pretraining done ({time.time() - t0:.0f}s)')
    slice_idx, slice_on = build_action_slices(keynet)
    log(f'action slice width={slice_idx.shape[1]}, codes={len(keynet.protos)}, retrieval={"off" if args.no_nn else "on"}')
    model = A2C_EPN(keynet, slice_idx, slice_on, mem_size=config['dict-len'], use_nn=not args.no_nn)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'])
    make_row = make_row_fn(slice_idx, slice_on)
    env = ArmEnv(config['n-workers'], SEED_TRAIN + args.seed)
    csvf = open(os.path.join(args.out, 'episodes.csv'), 'w', newline='')
    writer = csv.writer(csvf)
    writer.writerow(['episode', 'reward', 'targets_reached', 'mean25_reached', 'nearest_cosine', 'minutes'])
    reaches, t1 = [], time.time()

    def checkpoint(ep):                      # everything that training changes; the frozen keys are saved once in keys.pt
        os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
        torch.save({k: v.clone() for k, v in model.state_dict().items() if not k.startswith('keynet.')},
                   os.path.join(args.out, 'checkpoints', f'ep{ep:04d}.pt'))
    if args.checkpoint_every:
        checkpoint(0)
    for ep in range(args.episodes):
        reward, reached, cosine = run_episode(model, optimizer, env, make_row, train=True)
        reaches.append(reached)
        writer.writerow([ep + 1, reward, reached, float(np.mean(reaches[-25:])), cosine, (time.time() - t1) / 60])
        if (ep + 1) % 25 == 0:
            csvf.flush()
            log(f'episode {ep + 1}: targets reached avg25={np.mean(reaches[-25:]):.2f}, reward={reward:.2f}, '
                f'nearest_cos={cosine:.3f} ({(time.time() - t1) / 60:.1f} min)')
        if args.checkpoint_every and (ep + 1) % args.checkpoint_every == 0:
            checkpoint(ep + 1)
        if (ep + 1) % 100 == 0 or ep + 1 == args.episodes:
            torch.save({'model': model.state_dict(), 'slice_idx': slice_idx, 'slice_on': slice_on},
                       os.path.join(args.out, 'model.pt'))
    csvf.close()
    eval_env = ArmEnv(config['n-workers'], SEED_EVAL)
    torch.manual_seed(2021)
    with torch.no_grad():
        results = [run_episode(model, None, eval_env, make_row, train=False)[1] for _ in range(4)]
    log(f'held-out arms: targets reached per episode = {np.mean(results):.2f}')
    json.dump({'held_out_targets_reached': float(np.mean(results)), 'episodes': args.episodes},
              open(os.path.join(args.out, 'evaluation.json'), 'w'))
