# Break the arm. It keeps writing.

A robot arm writes a word in your browser. You break it while it writes: swap its wires, reverse a motor, make the
motors three times too strong, make each motor drag the next joint. Nothing retrains it. It tries a few moves, remembers
what each one did, and carries on.

**[Try it](https://randman444.github.io/meta-rl-draw/)** · By Chris Martinez ([LinkedIn](https://www.linkedin.com/in/chris-martinez-6749a9ba/))

The page runs the trained agent in plain JavaScript, no libraries. The training code is PyTorch and runs on a laptop CPU.

## How it works

The agent is a small reinforcement-learning agent (A2C) with an episodic memory of everything it has done on the
current arm. Before each move it asks, for every action it could take, "what happened the last time I did something
like this?" It finds the single nearest neighbor in its memory, using lookup keys that were trained once and then
frozen, and reads off what that action really did. A small transformer reads the memory as a whole. That is all: no
weight updates while it runs.

In symbols: every move $`(s_j, a_j)`$ of the current episode is stored with a key, a unit vector

```math
k(s, a) = \frac{f(g_a \odot s,\ a)}{\lVert f(g_a \odot s,\ a) \rVert},
```

where $`g_a`$ is a learned gate that keeps only the parts of the observation that matter for action $`a`$. Before each
move the agent scores every action $`a`$ against memory by cosine similarity and keeps one winner, with no averaging:

```math
j^\star(a) = \arg\max_{j} \ k(s_t, a)^\top k(s_j, a_j), \qquad \hat z_a = z_t + \big(\phi(s'_{j^\star}) - \phi(s_{j^\star})\big).
```

Here $`\phi`$ is a small learned encoder of the observation and $`z_t = \phi(s_t)`$, so $`\hat z_a`$ is "where I expect to
end up if I do $`a`$", built from what the most similar past move really did, and the
policy reads it for every action. The key network $`f`$ is trained once, before the agent learns to act, so that moves
with the same effect in the same episode get nearby keys. Then it is frozen, for two reasons: the $`\arg\max`$ passes no
gradient, so reinforcement learning could not train the keys anyway, and fixed keys mean a memory written at move 3
is still found the same way at move 140.

<details>
<summary><b>Learn more: what each symbol means</b></summary>

- **$`s`$, the situation.** What the agent sees right now: where the pen is, where the target is relative to the pen,
  and the three joint angles.
- **$`a`$, the action.** One of 12 buttons: which motor (3), which direction (2), small or big step (2).
- **$`g_a`$, the gate.** One number between 0 and 1 per feature of $`s`$, learned separately for each action. It answers
  "for this button, what in the situation matters?" For a given motor it can keep the joint angles and mostly ignore
  where the target is, because the target does not change what the motor does.
- **$`\odot`$** multiplies feature by feature, so $`g_a \odot s`$ is the situation with the irrelevant parts dimmed.
- **$`f`$, the key network.** A small neural network that turns the dimmed situation and the action into 32 numbers, a
  fingerprint for "doing $`a`$ in situation $`s`$".
- **Dividing by $`\lVert \cdot \rVert`$** scales the fingerprint to length 1 so that only its direction counts. Two
  fingerprints are compared by their dot product, the cosine similarity: 1 when they point the same way, about 0 when
  unrelated.
- **How $`f`$ learns, and why it is frozen.** Before the agent learns to act, it watches random flailing, and $`f`$ is
  trained so that two moves that did the same thing to the pen in the same episode get fingerprints pointing the same
  way. Then $`f`$ is locked, so a fingerprint means the same thing for the whole run.
- **$`j^\star`$, the lookup.** At move 40 the agent considers "motor 2, forward, big step", makes that fingerprint and finds
  the closest one in its memory of this episode. Say it is move 12: the same button from similar joint angles, when
  the pen went up and to the left.
- **$`\phi(s'_{j^\star}) - \phi(s_{j^\star})`$** is that "up and to the left", in the agent's own encoding $`\phi`$ of the
  observation. Added to the current encoding $`z_t`$, it gives $`\hat z_a`$: where the agent expects to end up if it
  presses that button now. If someone swapped the wires, move 12 already shows what motor 2 really does on this arm,
  which is why it copes without retraining.

</details>

It learned this on randomly rewired arms in 1,000 practice episodes. The arms on the page are damaged in ways it never
practiced on.

## Why it beats MAML

MAML and its first-order cousins such as Reptile adapt by taking gradient steps on the network's weights. That suits
tasks where what has to change is a smooth skill. A broken arm is different: what the agent needs is a few exact facts
("this motor turns that joint, backwards, twice as far"), and one move is enough to see each one. A gradient step
computed from a handful of noisy moves is a blurry place to write a fact down. A memory lookup keeps it exactly and
uses it on the very next move.

The measurements agree. MAML never learned the arm: after 4,000 training episodes its best setting touches under 2 targets
per episode, and the same network with the gradient steps switched off does slightly better (2.5). Our agent touches
about 10 after 1,000. A recurrent agent, which has to squeeze the facts into its hidden state, gets there in 4,000
episodes but falls behind once the damage is new.

## Results

Targets touched per 150-move episode on 128 arms nobody trained on, with no retraining:

| Damage | Ours, 1,000 episodes | Recurrent, 4,000 episodes | MAML, 4,000 episodes | Hand-written self-calibrating controller |
|---|---|---|---|---|
| Within the practiced range | 10.5 | 9.9 | 1.5 | 9.9 |
| Motors 2x too strong | 14.4 | 10.5 | 1.6 | 13.3 |
| Motors 3x too strong | 11.9 | 7.5 | 1.4 | 11.9 |
| Each motor drags the next joint, half | 8.9 | 7.8 | 2.0 | 7.0 |
| Each motor drags the next joint, fully | 8.5 | 6.7 | 2.4 | 7.4 |
| Jittery motors, 60% error | 7.4 | 5.9 | 1.4 | 6.6 |
| Motors at quarter strength | 3.5 | 2.6 | 0.6 | 4.6 |

A controller hand-written for this arm (try each motor once, remember what it did) is close to ours: ahead on weak
motors and plain strength changes, behind on drag, jitter and combined damage. Ours had to discover that strategy by
itself from practice alone. Caveats: it is a small 2D arm with one training run per baseline, and the MAML is our own
implementation with six settings tried. Everything else, including learning curves and harder arms, is in
[results/README.md](results/README.md).

## Run it

Open `web/index.html` in a browser. To train and test the agents yourself:

```bash
pip install -r python/requirements.txt
python python/arm_meta.py runs/original                                   # ours, about 1 hour on one core
python python/arm_rl2.py runs/recurrent --episodes 4000 --gamma 0.9 --entropy 0.01
python python/arm_maml.py runs/maml --episodes 4000
python python/arm_ood.py ours runs/original runs/original/keys.pt --temp 0.25 --seeds 31000 32000
```

## License

MIT, see `LICENSE`.
