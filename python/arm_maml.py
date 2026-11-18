"""A MAML baseline on the broken arm (Finn et al. 2017, with learned per-tensor step sizes as in Meta-SGD).

The policy is a feed-forward actor-critic with no memory. It adapts to an arm the only way MAML can: by gradient
steps on its own weights, taken inside the episode. Every `adapt_every` moves it takes one A2C gradient step on the
moves it has just made and carries on with the adapted weights. Meta-training backpropagates through those steps
(second order unless --first-order) so that the starting weights and the step sizes make the adaptation work.

Same arms, same 150-step episodes, 16 parallel arms and A2C losses as the other agents; the score counts the whole
episode, including the moves made before the first adaptation. --practice N is the more generous, original MAML setup:
N unscored practice episodes on each arm, with gradient steps, before the scored one. The other agents get none.

    python arm_maml.py <out_dir> [--episodes 4000] [--adapt-every 25] [--first-order] [--no-adapt] [--practice N]
"""
import argparse, csv, json, math, os, time
import numpy as np, torch
import torch.nn.functional as F
from arm_env import ArmEnv, N_ACTIONS, OBS_DIM, TRIAL_STEPS, EPISODE_STEPS
from arm_meta import config, SEED_TRAIN, SEED_EVAL

HIDDEN = 128
TEMP = 1.0          # test-time sampling temperature, set by the evaluation scripts


def init_params():
    def lin(o, i, scale=1.0):
        return torch.randn(o, i) * scale / math.sqrt(i), torch.zeros(o)
    p = {}
    p['W1'], p['b1'] = lin(HIDDEN, OBS_DIM + 1)
    p['W2'], p['b2'] = lin(HIDDEN, HIDDEN)
    p['Wp'], p['bp'] = lin(N_ACTIONS, HIDDEN, 0.1)
    p['Wv'], p['bv'] = lin(1, HIDDEN, 0.1)
    return {k: v.requires_grad_() for k, v in p.items()}


def forward(p, x):
    """p: weights with a leading per-arm dimension, x: [arms, features]."""
    h = F.elu(torch.einsum('boi,bi->bo', p['W1'], x) + p['b1'])
    h = F.elu(torch.einsum('boi,bi->bo', p['W2'], h) + p['b2'])
    return torch.einsum('boi,bi->bo', p['Wp'], h) + p['bp'], (torch.einsum('boi,bi->bo', p['Wv'], h) + p['bv']).squeeze(-1)


def a2c_loss(seg, bootstrap, hp):
    """One A2C loss per arm for a segment of moves."""
    G, targets = bootstrap, []
    for r, m in zip(seg['r'][::-1], seg['mask'][::-1]):
        G = r + hp['gamma'] * G * m
        targets.append(G)
    target = torch.stack(targets[::-1])                                 # [T, arms]
    value, logp, ent = torch.stack(seg['v']), torch.stack(seg['logp']), torch.stack(seg['ent'])
    adv = (target - value).detach()
    return (0.5 * F.smooth_l1_loss(value, target, reduction='none') - logp * adv - hp['entropy'] * ent).mean(0)


def same_arm_reset(env):
    """A new episode (new pose, new targets) on the arms the last reset() drew."""
    sample, env._sample_arm = env._sample_arm, lambda: None
    try:
        return env.reset()
    finally:
        env._sample_arm = sample


def run_episode(theta, log_alpha, optimizer, env, hp, train=True):
    """One scored episode per arm, after hp['practice'] unscored practice episodes on the same arm.

    A gradient step per arm follows every `adapt_every` moves, practice or scored. With no practice the meta-objective is
    the loss over the whole scored episode (do well while adapting); with practice it is the scored episode only, as in
    the original MAML setup, so the practice episodes are free to explore.
    """
    n, practice = env.n, hp.get('practice', 0)
    T = lambda x: torch.as_tensor(np.asarray(x)).float()
    obs = env.reset()
    fast = {k: v.unsqueeze(0).expand(n, *v.shape) for k, v in theta.items()}     # every arm starts from the same weights
    outer = 0.0
    inp = lambda: T(np.concatenate([obs, (env.trial_t / TRIAL_STEPS)[:, None]], 1))
    with torch.enable_grad():
        for ep in range(practice + 1):
            scored = ep == practice
            if ep:
                obs = same_arm_reset(env)
            reach_sum, done = np.zeros(n, np.float32), np.zeros(n, bool)
            while not done.all():
                seg = {'r': [], 'mask': [], 'v': [], 'logp': [], 'ent': []}
                for _ in range(hp['adapt_every']):
                    logits, value = forward(fast, inp())
                    lsm = F.log_softmax(logits, -1)
                    action = torch.distributions.Categorical(logits=logits.detach() / TEMP).sample() if TEMP else logits.argmax(-1)
                    obs, rewards, done, new_trial, settled, reached = env.step(action.numpy())
                    reach_sum += reached
                    seg['r'].append(T(rewards)); seg['mask'].append(T(1. - done.astype(np.float32)))
                    seg['v'].append(value); seg['logp'].append(lsm.gather(1, action[:, None]).squeeze(-1)); seg['ent'].append(-(lsm * lsm.exp()).sum(-1))
                    if done.all():
                        break
                with torch.no_grad():
                    bootstrap = forward(fast, inp())[1]
                loss = a2c_loss(seg, bootstrap, hp)                               # [arms]
                if scored or not practice:
                    outer = outer + loss.mean() * len(seg['r']) / EPISODE_STEPS
                if not (scored and done.all()) and not hp['no_adapt']:            # one gradient step per arm on its own moves
                    names = list(fast)
                    grads = torch.autograd.grad(loss.sum(), [fast[k] for k in names], create_graph=train and not hp['first_order'], retain_graph=train)
                    fast = {k: fast[k] - log_alpha[k].exp() * g for k, g in zip(names, grads)}
                if not train:
                    fast = {k: v.detach().requires_grad_() for k, v in fast.items()}
        if train:
            optimizer.zero_grad(); outer.backward()
            torch.nn.utils.clip_grad_norm_(list(theta.values()) + list(log_alpha.values()), 10.0)
            optimizer.step()
    return float(reach_sum.mean())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--episodes', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=config['lr'])
    ap.add_argument('--alpha', type=float, default=0.05, help='initial inner step size (learned afterwards)')
    ap.add_argument('--adapt-every', type=int, default=25)
    ap.add_argument('--first-order', action='store_true')
    ap.add_argument('--practice', type=int, default=0, help='unscored practice episodes on each arm before the scored one')
    ap.add_argument('--resume', default=None, help='a model-<episode>.pt saved by an interrupted run of this out_dir; training carries on from it')
    ap.add_argument('--no-adapt', action='store_true', help='reference: the same policy with no gradient steps inside the episode')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); torch.set_num_threads(int(os.environ.get('THREADS', 1)))
    hp = {'gamma': args.gamma, 'entropy': args.entropy, 'adapt_every': args.adapt_every, 'first_order': args.first_order, 'no_adapt': args.no_adapt, 'practice': args.practice}
    theta = init_params()
    log_alpha = {k: torch.full((), math.log(args.alpha), requires_grad=True) for k in theta}
    start = 0
    if args.resume:                 # the optimizer state is not saved, so Adam starts its averages again
        ck = torch.load(args.resume, weights_only=False)
        theta = {k: v.requires_grad_() for k, v in ck['theta'].items()}
        log_alpha = {k: v.requires_grad_() for k, v in ck['log_alpha'].items()}
        start = int(os.path.basename(args.resume).split('-')[1].split('.')[0])
    optimizer = torch.optim.Adam(list(theta.values()) + list(log_alpha.values()), lr=args.lr)
    env = ArmEnv(config['n-workers'], SEED_TRAIN + args.seed + start)
    logf = open(os.path.join(args.out, 'log.txt'), 'a')
    csvf = open(os.path.join(args.out, 'episodes.csv'), 'a' if start else 'w', newline='')
    writer = csv.writer(csvf)
    if not start:
        writer.writerow(['episode', 'targets_reached', 'mean25_reached', 'minutes'])
    reaches, t1 = [], time.time()
    for ep in range(start, args.episodes):
        reaches.append(run_episode(theta, log_alpha, optimizer, env, hp))
        writer.writerow([ep + 1, reaches[-1], float(np.mean(reaches[-25:])), (time.time() - t1) / 60])
        if (ep + 1) % 100 == 0:
            csvf.flush()
            msg = (f'episode {ep + 1}: targets reached avg25={np.mean(reaches[-25:]):.2f}, '
                   f'inner step size {np.mean([v.exp().item() for v in log_alpha.values()]):.3f} ({(time.time() - t1) / 60:.1f} min)')
            print(msg, flush=True); logf.write(msg + '\n'); logf.flush()
        if (ep + 1) % 500 == 0 or ep + 1 == args.episodes:
            torch.save({'theta': {k: v.detach() for k, v in theta.items()}, 'log_alpha': {k: v.detach() for k, v in log_alpha.items()}, 'hp': hp},
                       os.path.join(args.out, f'model-{ep + 1}.pt'))
    csvf.close()
    eval_env = ArmEnv(config['n-workers'], SEED_EVAL)
    torch.manual_seed(2021)
    results = [run_episode(theta, log_alpha, None, eval_env, hp, train=False) for _ in range(4)]
    msg = f'held-out arms: targets reached per episode = {np.mean(results):.2f}'
    print(msg, flush=True); logf.write(msg + '\n')
    json.dump({'held_out_targets_reached': float(np.mean(results)), 'episodes': args.episodes, 'args': vars(args)},
              open(os.path.join(args.out, 'evaluation.json'), 'w'))
