"""Export a trained recurrent baseline (arm_rl2.py) for the web page.

    python export_rl2.py <model.pt> <name> <out_dir> [trained_episodes]

Writes <name>.js (weights as float16) and <name>.trace.json (60 steps of PyTorch inputs and logits, so
web/check_recurrent.js can confirm the JavaScript port gives the same numbers).
"""
import base64, json, sys
import numpy as np, torch
from arm_env import ArmEnv, N_ACTIONS, OBS_DIM, TRIAL_STEPS
from arm_rl2 import RL2

model_path, name, out_dir = sys.argv[1:4]
episodes = int(sys.argv[4]) if len(sys.argv) > 4 else None
model = RL2(); model.load_state_dict(torch.load(model_path, weights_only=False)); model.eval()
half = {k: v.numpy().astype(np.float16) for k, v in model.state_dict().items() if not k.startswith('critic')}
tensors = {k: {'shape': list(v.shape), 'h16': base64.b64encode(np.ascontiguousarray(v).tobytes()).decode()} for k, v in half.items()}
pack = {'name': name, 'actions': N_ACTIONS, 'obs': OBS_DIM, 'hidden': model.hidden, 'trained_episodes': episodes, 'tensors': tensors}
with open(f'{out_dir}/{name}.js', 'w') as f:
    f.write(f'(typeof window !== "undefined" ? window : globalThis).ARM_RIVALS = Object.assign((typeof window !== "undefined" ? window : globalThis).ARM_RIVALS || {{}}, {{{json.dumps(name)}: {json.dumps(pack)}}});\n')

# trace with the float16 weights the page will use
model.load_state_dict({**model.state_dict(), **{k: torch.as_tensor(v.astype(np.float32)) for k, v in half.items()}})
torch.manual_seed(0)
env = ArmEnv(1, 123); obs = env.reset()
hc = (torch.zeros(1, model.hidden), torch.zeros(1, model.hidden))
prev_a, prev_r, steps = np.zeros((1, N_ACTIONS), np.float32), np.zeros((1, 1), np.float32), []
with torch.no_grad():
    for _ in range(60):
        x = torch.as_tensor(np.concatenate([obs, (env.trial_t / TRIAL_STEPS)[:, None], prev_a, prev_r], 1)).float()
        logits, _, hc = model(x, hc)
        action = int(torch.distributions.Categorical(logits=logits).sample())
        trial_t = float(env.trial_t[0])
        step_obs = obs[0].tolist()
        obs, rewards, done, new_trial, settled, reached = env.step(np.array([action]))
        steps.append({'obs': step_obs, 'trial_t': trial_t, 'logits': logits[0].tolist(), 'action': action, 'reward': float(rewards[0])})
        prev_a = np.eye(N_ACTIONS, dtype=np.float32)[[action]]; prev_r = rewards[:, None].astype(np.float32)
json.dump({'steps': steps}, open(f'{out_dir}/{name}.trace.json', 'w'))
print(f'{name} exported: {sum(v.size for v in half.values())} numbers')
