"""Broken robot arm: a batch of planar 3-joint arms whose wiring is different, and hidden, every episode.

Hidden per episode (the "chemistry"):
  * link lengths
  * which motor drives which joint
  * each motor's strength and direction (it may be reversed)
  * possibly one dead motor

12 actions: motor (3) x direction (2) x step size (2). The agent sees the pen position, the
target relative to the pen and the joint angles (the time spent on this target is given separately). It never
sees lengths, wiring or strengths. An episode is 150 steps; each target must be reached
within 20 steps, then a new one appears (a "trial", as in Alchemy).

Reward: progress towards the target, plus 1 for touching it.
"""
import os
import numpy as np

# Difficulty dials, set through the environment so every script sees the same task:
#   ARM_WIRING=motors (default): 3 motors x 2 directions x 2 step sizes, hidden motor->joint map and strengths.
#   ARM_WIRING=board: a miswired control board. Every button drives one hidden joint by one hidden signed
#                     step; some buttons are dead. Nothing about a button can be guessed from its number.
#   ARM_JOINTS: number of joints (default 3).   ARM_BUTTONS: number of buttons for the board (default 4 x joints).
WIRING = os.environ.get('ARM_WIRING', 'motors')
N_JOINTS = int(os.environ.get('ARM_JOINTS', 3))
N_ACTIONS = int(os.environ.get('ARM_BUTTONS', 4 * N_JOINTS)) if WIRING == 'board' else 4 * N_JOINTS
OBS_DIM = 5 + 2 * N_JOINTS
LENGTH_SCALE = 3.0 / N_JOINTS     # keeps the total reach the same when joints are added
EPISODE_STEPS = 150
TRIAL_STEPS = 20
REACH = 0.06
STEP_SIZES = (0.07, 0.21)        # radians, before the hidden motor strength is applied
LIMIT = 2.6                      # joint limit for joints 2 and 3 (relative angle)


class ArmEnv:
    def __init__(self, n, seed, breakage=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.breakage = breakage or {}

    # ----- hidden episode parameters -----
    def _sample_arm(self):
        n, rng = self.n, self.rng
        self.lengths = rng.uniform(0.2, 0.5, (n, N_JOINTS)) * LENGTH_SCALE
        self.perm = np.stack([rng.permutation(N_JOINTS) for _ in range(n)])       # motor -> joint
        self.gain = rng.uniform(0.5, 1.5, (n, N_JOINTS)) * rng.choice([-1.0, 1.0], (n, N_JOINTS))
        dead = rng.random(n) < 0.25
        self.gain[np.arange(n)[dead], rng.integers(0, N_JOINTS, dead.sum())] = 0.0
        if WIRING == 'board':
            A, J = N_ACTIONS, N_JOINTS
            assert A >= 2 * J, 'the board needs at least two buttons per joint'
            joint = np.concatenate([np.tile(np.arange(J), (n, 2)), rng.integers(0, J, (n, A - 2 * J))], 1)
            sign = np.concatenate([np.ones((n, J)), -np.ones((n, J)), rng.choice([-1.0, 1.0], (n, A - 2 * J))], 1)
            step = rng.uniform(0.04, 0.25, (n, A)) * sign
            step[:, 2 * J:] *= rng.random((n, A - 2 * J)) >= 0.15          # dead buttons, never the guaranteed pair
            order = np.argsort(rng.random((n, A)), 1)                       # shuffle which button is which
            self.btn_joint = np.take_along_axis(joint, order, 1)
            self.btn_step = np.take_along_axis(step, order, 1)
        for key, value in self.breakage.items():                                  # demo sliders override
            getattr(self, key)[:] = value

    def pen(self, theta=None, lengths=None):
        theta = self.theta if theta is None else theta
        lengths = self.lengths if lengths is None else lengths
        cum = np.cumsum(theta, -1)
        return np.stack([(lengths * np.cos(cum)).sum(-1), (lengths * np.sin(cum)).sum(-1)], -1)

    def _random_pose(self, n):
        th = self.rng.uniform(-2.2, 2.2, (n, N_JOINTS))
        th[:, 0] = self.rng.uniform(-np.pi, np.pi, n)
        return th

    def _new_targets(self, which):
        """A reachable target: the pen position of a random nearby pose of this very arm."""
        k = int(which.sum())
        if k == 0:
            return
        best, best_d = None, None
        for _ in range(4):                                            # prefer targets 0.15..0.6 away
            th = self.theta[which] + self.rng.uniform(-1.2, 1.2, (k, N_JOINTS))
            th[:, 1:] = np.clip(th[:, 1:], -LIMIT, LIMIT)
            cand = self.pen(th, self.lengths[which])
            dist = np.linalg.norm(cand - self.pen()[which], axis=-1)
            score = np.abs(dist - 0.35)
            if best is None:
                best, best_d = cand, score
            else:
                better = score < best_d
                best[better], best_d[better] = cand[better], score[better]
        self.target[which] = best
        self.trial_t[which] = 0

    def obs(self):
        p = self.pen()
        rel = self.target - p
        return np.concatenate([p, rel, np.linalg.norm(rel, axis=-1, keepdims=True),
                               np.sin(self.theta), np.cos(self.theta)], -1).astype(np.float32)

    def reset(self):
        self._sample_arm()
        self.theta = self._random_pose(self.n)
        self.target = np.zeros((self.n, 2))
        self.trial_t = np.zeros(self.n)
        self.t = 0
        self._new_targets(np.ones(self.n, bool))
        return self.obs()

    def effect(self, actions, theta=None):
        """Joint angles after the action (hidden wiring applied)."""
        theta = (self.theta if theta is None else theta).copy()
        idx = np.arange(len(theta))
        if WIRING == 'board':
            theta[idx, self.btn_joint[idx, actions]] += self.btn_step[idx, actions]
            theta[:, 1:] = np.clip(theta[:, 1:], -LIMIT, LIMIT)
            return theta
        motor, direction, size = actions // 4, np.where(actions % 4 < 2, 1.0, -1.0), np.take(STEP_SIZES, actions % 2)
        joint = self.perm[idx, motor]
        theta[idx, joint] += direction * size * self.gain[idx, motor]
        theta[:, 1:] = np.clip(theta[:, 1:], -LIMIT, LIMIT)
        return theta

    def step(self, actions):
        """Returns (obs, reward, done, new_trial, obs_before_target_change)."""
        actions = np.asarray(actions)
        d0 = np.linalg.norm(self.target - self.pen(), axis=-1)
        self.theta = self.effect(actions)
        d1 = np.linalg.norm(self.target - self.pen(), axis=-1)
        reached = d1 < REACH
        reward = 2.0 * (d0 - d1) + reached.astype(np.float64)
        self.trial_t += 1
        self.t += 1
        settled = self.obs()                                          # same target: a clean transition for memory
        new_trial = reached | (self.trial_t >= TRIAL_STEPS)
        self._new_targets(new_trial)
        done = np.full(self.n, self.t >= EPISODE_STEPS)
        return self.obs(), reward.astype(np.float32), done, new_trial, settled, reached


# ----- references that need no learning -----
def run_reference(kind, n=64, seed=30_000, episodes=2):
    """perfect: knows the hidden wiring and takes the best single step.
    calibrate: classical online calibration. It remembers the last pen displacement each action
               produced, tries every action it has not tried yet, then takes the best predicted step.
    random: uniform random actions."""
    env = ArmEnv(n, seed)
    rng = np.random.default_rng(seed + 1)
    totals, reaches = [], []
    for _ in range(episodes):
        env.reset()
        total, reach = np.zeros(n), np.zeros(n)
        last = np.full((n, N_ACTIONS, 2), np.nan)
        for _ in range(EPISODE_STEPS):
            p = env.pen()
            if kind == "random":
                a = rng.integers(0, N_ACTIONS, n)
            else:
                pred = np.zeros((n, N_ACTIONS))
                for act in range(N_ACTIONS):
                    if kind == "perfect":
                        landing = env.pen(env.effect(np.full(n, act)))
                    else:
                        landing = p + np.nan_to_num(last[:, act])
                    pred[:, act] = np.linalg.norm(env.target - landing, axis=-1)
                a = pred.argmin(1)
                if kind == "calibrate":
                    untried = np.isnan(last[..., 0])
                    first = untried.argmax(1)
                    a = np.where(untried.any(1), first, a)
            _, r, _, _, _, reached = env.step(a)
            if kind == "calibrate":
                last[np.arange(n), a] = env.pen() - p
            total += r; reach += reached
        totals.append(total.mean()); reaches.append(reach.mean())
    return float(np.mean(totals)), float(np.mean(reaches))


if __name__ == "__main__":
    print(f"wiring={WIRING} joints={N_JOINTS} buttons={N_ACTIONS}")
    for kind in ("random", "calibrate", "perfect"):
        r, k = run_reference(kind)
        print(f"{kind:10s} reward per episode {r:6.2f}   targets reached per episode {k:5.2f}")
