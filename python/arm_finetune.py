"""Does it help to keep training the trained agent on one fixed arm?

Loads a trained retrieval agent, fixes the arm (every parallel worker gets the same damage, different targets),
scores it, keeps training with the same A2C settings, and scores it again, on that arm and on fresh random arms.

    python arm_finetune.py <run_dir> <keys.pt> <out_dir> <arm_name> [--episodes 100]
"""
import argparse, json, os, time
import numpy as np, torch
from torch import optim
import arm_env
from arm_env import ArmEnv
from arm_meta import SituationKey, A2C_EPN, build_action_slices, make_row_fn, run_episode, config, SEED_EVAL

ARMS = {   # training range: |strength| 0.5 to 1.5, at most one dead motor
    'in-range':     {'gain': [1.2, -0.7, 0.9], 'perm': [1, 2, 0], 'lengths': [0.30, 0.40, 0.35]},
    'weak-motors':  {'gain': [0.2, -0.15, 0.3], 'perm': [1, 2, 0], 'lengths': [0.30, 0.40, 0.35]},
    'one-good-two-weak': {'gain': [0.1, 1.5, -0.25], 'perm': [2, 0, 1], 'lengths': [0.45, 0.25, 0.30]},
    'two-dead':     {'gain': [0.0, 1.0, 0.0], 'perm': [0, 1, 2], 'lengths': [0.30, 0.40, 0.35]},
}


def reference(kind, breakage, n=64, episodes=2, seed=30_000):
    """Same as arm_env.run_reference, on a fixed arm."""
    from arm_env import N_ACTIONS, EPISODE_STEPS
    env, rng = ArmEnv(n, seed, breakage), np.random.default_rng(seed + 1)
    reaches = []
    for _ in range(episodes):
        env.reset()
        reach, last = np.zeros(n), np.full((n, N_ACTIONS, 2), np.nan)
        for _ in range(EPISODE_STEPS):
            p = env.pen()
            pred = np.zeros((n, N_ACTIONS))
            for act in range(N_ACTIONS):
                landing = env.pen(env.effect(np.full(n, act))) if kind == 'perfect' else p + np.nan_to_num(last[:, act])
                pred[:, act] = np.linalg.norm(env.target - landing, axis=-1)
            a = pred.argmin(1)
            if kind == 'calibrate':
                untried = np.isnan(last[..., 0])
                a = np.where(untried.any(1), untried.argmax(1), a)
            reached = env.step(a)[-1]
            if kind == 'calibrate':
                last[np.arange(n), a] = env.pen() - p
            reach += reached
        reaches.append(reach.mean())
    return float(np.mean(reaches))


def score(model, make_row, breakage, seed, episodes=4):
    env = ArmEnv(config['n-workers'], seed, breakage)
    with torch.no_grad():
        return float(np.mean([run_episode(model, None, env, make_row, train=False)[1] for _ in range(episodes)]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('keys'); ap.add_argument('out'); ap.add_argument('arm')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--lr', type=float, default=config['lr'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(int(os.environ.get('THREADS', 1)))
    breakage = {k: np.asarray(v, float) if k != 'perm' else np.asarray(v) for k, v in ARMS[args.arm].items()}
    ck = torch.load(args.keys, weights_only=False)
    keynet = SituationKey(); keynet.load_state_dict(ck['net']); keynet.protos = ck['protos']; keynet.eval()
    slice_idx, slice_on = build_action_slices(keynet)
    model = A2C_EPN(keynet, slice_idx, slice_on, mem_size=config['dict-len'])
    model.load_state_dict(torch.load(os.path.join(args.run, 'model.pt'), weights_only=False)['model'])
    make_row = make_row_fn(slice_idx, slice_on)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    out = {'arm': args.arm, 'lr': args.lr, 'classical': reference('calibrate', breakage), 'told_the_wiring': reference('perfect', breakage)}
    out['before_this_arm'] = score(model, make_row, breakage, 40_000)
    out['before_random_arms'] = score(model, make_row, None, SEED_EVAL)
    print(json.dumps(out), flush=True)
    env, curve, t0 = ArmEnv(config['n-workers'], 50_000, breakage), [], time.time()
    for ep in range(args.episodes):
        curve.append(run_episode(model, optimizer, env, make_row, train=True)[1])
        if (ep + 1) % 10 == 0:
            print(f'episode {ep + 1}: mean of last 10 = {np.mean(curve[-10:]):.2f} ({time.time() - t0:.0f}s)', flush=True)
        if (ep + 1) in (25, 50, 100, 200):
            out[f'after_{ep + 1}_this_arm'] = score(model, make_row, breakage, 40_000)
            out[f'after_{ep + 1}_random_arms'] = score(model, make_row, None, SEED_EVAL)
            print(json.dumps(out), flush=True)
    out['curve'] = curve
    json.dump(out, open(os.path.join(args.out, f'{args.arm}.json'), 'w'))
