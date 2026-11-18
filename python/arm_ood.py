"""Unseen damage, no retraining: how does a trained agent score on arms outside its training range?

Training range: every motor has strength 0.5 to 1.5 (either direction), drives exactly one joint, at most one is dead.
Here: motor strength swept from 0.1 to 3, motors that also drag a second joint (crosstalk), and jittery motors.

    python arm_ood.py ours <run_dir> <keys.pt> [--temp 0.25]     python arm_ood.py rl2 <model.pt> [--temp 0.25]
    python arm_ood.py maml <model.pt> [--temp 0.25]              python arm_ood.py refs
"""
import json, sys
import numpy as np, torch
import arm_env
from arm_env import ArmEnv, N_ACTIONS, N_JOINTS, EPISODE_STEPS, STEP_SIZES, LIMIT

SIGNS = np.array([1.0, -1.0, 1.0])


class OddArm(ArmEnv):
    """ArmEnv plus damage the agents never trained on."""
    def __init__(self, n, seed, strength=None, crosstalk=0.0, jitter=0.0):
        super().__init__(n, seed, None if strength is None else {'gain': SIGNS * strength})
        self.crosstalk, self.jitter, self._live = crosstalk, jitter, False
        self.noise_rng = np.random.default_rng(seed + 7)

    def effect(self, actions, theta=None):
        theta = (self.theta if theta is None else theta).copy()
        idx = np.arange(len(theta))
        motor, direction, size = actions // 4, np.where(actions % 4 < 2, 1.0, -1.0), np.take(STEP_SIZES, actions % 2)
        joint = self.perm[idx, motor]
        move = direction * size * self.gain[idx, motor]
        if self._live and self.jitter:                                   # only real moves are noisy, predictions are not
            move = move * (1 + self.jitter * self.noise_rng.standard_normal(len(theta)))
        theta[idx, joint] += move
        if self.crosstalk:                                               # the motor also drags the next joint along
            theta[idx, (joint + 1) % N_JOINTS] += self.crosstalk * move
        theta[:, 1:] = np.clip(theta[:, 1:], -LIMIT, LIMIT)
        return theta

    def step(self, actions):
        self._live = True
        try:
            return super().step(actions)
        finally:
            self._live = False


CASES = [('trained range', {})] + [(f'strength {g}', {'strength': g}) for g in (3.0, 2.0, 1.0, 0.5, 0.35, 0.25, 0.15, 0.1)] \
      + [(f'crosstalk {c}', {'crosstalk': c}) for c in (0.5, 1.0, -1.0)] + [(f'jitter {j}', {'jitter': j}) for j in (0.3, 0.6)]


def reference(kind, case, n=64, seed=30_000):
    env, rng = OddArm(n, seed, **case), np.random.default_rng(seed + 1)
    env.reset()
    reach, last = np.zeros(n), np.full((n, N_ACTIONS, 2), np.nan)
    for _ in range(EPISODE_STEPS):
        p = env.pen()
        if kind == 'random':
            a = rng.integers(0, N_ACTIONS, n)
        else:
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
    return float(reach.mean())


class Sharp:
    """Stands in for torch's Categorical: samples at a lower temperature (0 = always the top action)."""
    T = 1.0

    def __init__(self, probs=None, logits=None):
        self.logits = probs.log() if logits is None else logits

    def sample(self):
        if Sharp.T == 0:
            return self.logits.argmax(-1)
        return torch.distributions.Categorical(logits=self.logits / Sharp.T).sample()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('who', choices=['refs', 'ours', 'rl2', 'maml']); ap.add_argument('paths', nargs='*')
    ap.add_argument('--temp', type=float, default=1.0, help='sampling temperature at test time (training always uses 1)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[30_000])
    ap.add_argument('--out', default='/dev/null')
    args = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(1)
    Sharp.T = args.temp
    out = {}
    if args.who == 'refs':
        for name, case in CASES:
            out[name] = {k: float(np.mean([reference(k, case, seed=s) for s in args.seeds])) for k in ('random', 'calibrate', 'perfect')}
            print(name, json.dumps(out[name]), flush=True)
    elif args.who == 'ours':
        import arm_meta
        from arm_meta import SituationKey, A2C_EPN, build_action_slices, make_row_fn, run_episode, config
        arm_meta.Categorical = Sharp
        ck = torch.load(args.paths[1], weights_only=False)
        keynet = SituationKey(); keynet.load_state_dict(ck['net']); keynet.protos = ck['protos']; keynet.eval()
        slice_idx, slice_on = build_action_slices(keynet)
        model = A2C_EPN(keynet, slice_idx, slice_on, mem_size=config['dict-len'])
        model.load_state_dict(torch.load(args.paths[0] + '/model.pt', weights_only=False)['model'])
        make_row = make_row_fn(slice_idx, slice_on)
        for name, case in CASES:
            with torch.no_grad():
                out[name] = float(np.mean([run_episode(model, None, OddArm(64, s, **case), make_row, train=False)[1] for s in args.seeds]))
            print(name, round(out[name], 2), flush=True)
    elif args.who == 'maml':
        import arm_maml
        arm_maml.TEMP = args.temp
        ck = torch.load(args.paths[0], weights_only=False)
        theta = {k: v.requires_grad_() for k, v in ck['theta'].items()}        # the inner gradient steps still run at test time
        for name, case in CASES:
            out[name] = float(np.mean([arm_maml.run_episode(theta, ck['log_alpha'], None, OddArm(64, s, **case), ck['hp'], train=False) for s in args.seeds]))
            print(name, round(out[name], 2), flush=True)
    else:
        import arm_rl2
        from arm_rl2 import RL2, run_episode
        arm_rl2.Categorical = Sharp
        model = RL2(); model.load_state_dict(torch.load(args.paths[0], weights_only=False))
        for name, case in CASES:
            with torch.no_grad():
                out[name] = float(np.mean([run_episode(model, None, OddArm(64, s, **case), {'gamma': 0.9, 'entropy': 0.01}, train=False)[1] for s in args.seeds]))
            print(name, round(out[name], 2), flush=True)
    json.dump(out, open(args.out, 'w'))
