"""Standard recurrent meta-RL baseline (RL^2 style: Duan et al. 2016, Wang et al. 2016) on the same arm task.

An LSTM sees the observation, its previous action and the previous reward, and keeps its hidden
state for the whole episode, so anything it learns about the hidden wiring has to live in that
state. Same environment, same A2C loss, same 16 workers and 20-step updates as arm_meta.py.

  python arm_rl2.py <out_dir> [--episodes 1000] [--seed 0] [--gamma 0.7] [--lr 7.5e-4] [--entropy 0.05] [--hidden 256]
"""
import argparse, csv, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from arm_env import ArmEnv, N_ACTIONS, OBS_DIM, TRIAL_STEPS
from arm_meta import config, SEED_TRAIN, SEED_EVAL


class RL2(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM + 1 + N_ACTIONS + 1, 128), nn.ELU())
        self.lstm = nn.LSTMCell(128, hidden)
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x, state):
        h, c = self.lstm(self.encoder(x), state)
        return self.actor(h), self.critic(h), (h, c)


def run_episode(model, optimizer, env, hp, train=True):
    n = env.n
    T = lambda x: torch.as_tensor(np.asarray(x)).float()
    obs = env.reset()
    hc = (torch.zeros(n, model.hidden), torch.zeros(n, model.hidden))
    prev_a, prev_r = np.zeros((n, N_ACTIONS), np.float32), np.zeros((n, 1), np.float32)
    reward_sum, reach_sum, done = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, bool)

    def inp():
        return T(np.concatenate([obs, (env.trial_t / TRIAL_STEPS)[:, None], prev_a, prev_r], 1))

    while not done.all():
        rewards_list, masks_list, values, log_probs, entropies = [], [], [], [], []
        hc = (hc[0].detach(), hc[1].detach())
        for _ in range(config['n-step-update']):
            with torch.set_grad_enabled(train):
                logits, value, hc = model(inp(), hc)
            lsm = F.log_softmax(logits, -1)
            action = Categorical(logits=logits).sample()
            actions = action.numpy()
            obs, rewards, done, new_trial, settled, reached = env.step(actions)
            reward_sum += rewards; reach_sum += reached
            prev_a = np.eye(N_ACTIONS, dtype=np.float32)[actions]; prev_r = rewards[:, None].astype(np.float32)
            rewards_list.append(rewards); masks_list.append(1. - done.astype(np.float32))
            values.append(value.squeeze(-1)); log_probs.append(lsm.gather(1, action[:, None]).squeeze(-1))
            entropies.append(-(lsm * lsm.exp()).sum(-1))
            if done.all():
                break
        if not train:
            continue
        with torch.no_grad():
            G = model(inp(), hc)[1].squeeze(-1).numpy()
        targets = []
        for r, mask in zip(rewards_list[::-1], masks_list[::-1]):
            G = r + hp['gamma'] * G * mask
            targets.append(G)
        td_target = torch.as_tensor(np.stack(targets[::-1])).float().reshape(-1)
        v = torch.stack(values).reshape(-1)
        advantage = (td_target - v).detach()
        loss = (config['value-loss-weight'] * F.smooth_l1_loss(v, td_target)
                - (torch.stack(log_probs).reshape(-1) * advantage).mean()
                - hp['entropy'] * torch.stack(entropies).mean())
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad-clip-norm'])
        optimizer.step()
    return float(reward_sum.mean()), float(reach_sum.mean())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--episodes', type=int, default=config['n-episodes'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gamma', type=float, default=config['gamma'])
    ap.add_argument('--lr', type=float, default=config['lr'])
    ap.add_argument('--entropy', type=float, default=config['entropy-weight'])
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--arm', help='train on one fixed arm from arm_finetune.ARMS instead of a new random arm every episode')
    ap.add_argument('--init', help='start from the weights of an earlier run (its model.pt)')
    ap.add_argument('--checkpoint-every', type=int, default=0, help='also save the weights every N episodes (for the web replay)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('THREADS', 1)))
    hp = {'gamma': args.gamma, 'entropy': args.entropy}
    model = RL2(args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    breakage = None
    if args.arm:
        from arm_finetune import ARMS
        breakage = {k: np.asarray(v, float) if k != 'perm' else np.asarray(v) for k, v in ARMS[args.arm].items()}
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=False))
    env = ArmEnv(config['n-workers'], (50_000 if args.arm else SEED_TRAIN) + args.seed, breakage)

    def score(br, seed):
        e = ArmEnv(config['n-workers'], seed, br)
        with torch.no_grad():
            return float(np.mean([run_episode(model, None, e, hp, train=False)[1] for _ in range(4)]))
    if args.arm:
        print(json.dumps({'arm': args.arm, 'before_this_arm': score(breakage, 40_000), 'before_random_arms': score(None, SEED_EVAL)}), flush=True)
    logf = open(os.path.join(args.out, 'log.txt'), 'a')
    csvf = open(os.path.join(args.out, 'episodes.csv'), 'w', newline='')
    writer = csv.writer(csvf)
    writer.writerow(['episode', 'reward', 'targets_reached', 'mean25_reached', 'nearest_cosine', 'minutes'])
    reaches, t1 = [], time.time()

    def checkpoint(ep):
        os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
        torch.save({k: v.clone() for k, v in model.state_dict().items()}, os.path.join(args.out, 'checkpoints', f'ep{ep:04d}.pt'))
    if args.checkpoint_every:
        checkpoint(0)
    for ep in range(args.episodes):
        reward, reached = run_episode(model, optimizer, env, hp)
        if args.checkpoint_every and (ep + 1) % args.checkpoint_every == 0:
            checkpoint(ep + 1)
        reaches.append(reached)
        writer.writerow([ep + 1, reward, reached, float(np.mean(reaches[-25:])), 0, (time.time() - t1) / 60])
        if args.arm and (ep + 1) in (25, 50, 100, 200, 300, 500, 1000):
            print(json.dumps({'arm': args.arm, 'episode': ep + 1, 'this_arm': score(breakage, 40_000), 'random_arms': score(None, SEED_EVAL)}), flush=True)
        if (ep + 1) % 50 == 0:
            csvf.flush()
            msg = f'episode {ep + 1}: targets reached avg25={np.mean(reaches[-25:]):.2f}, reward={reward:.2f} ({(time.time() - t1) / 60:.1f} min)'
            print(msg, flush=True); logf.write(msg + '\n'); logf.flush()
    csvf.close()
    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    eval_env = ArmEnv(config['n-workers'], SEED_EVAL)
    torch.manual_seed(2021)
    with torch.no_grad():
        results = [run_episode(model, None, eval_env, hp, train=False)[1] for _ in range(4)]
    msg = f'held-out arms: targets reached per episode = {np.mean(results):.2f}'
    print(msg, flush=True); logf.write(msg + '\n')
    json.dump({'held_out_targets_reached': float(np.mean(results)), 'episodes': args.episodes, 'args': vars(args)},
              open(os.path.join(args.out, 'evaluation.json'), 'w'))
