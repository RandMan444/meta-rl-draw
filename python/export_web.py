"""Export a trained agent for the web page, plus a short trace the JavaScript port is checked against.

  [ARM_WIRING=board ...] python export_web.py <run_dir> <keys.pt> <name> <out_dir>

Writes <out_dir>/<name>.js (weights as base64 float32, loaded with a plain <script> tag so the page also works
from file://) and <out_dir>/<name>.trace.json (observations, actions and the PyTorch logits for 60 steps).
"""
import base64, json, sys
import numpy as np
import torch

import arm_meta as m
from arm_env import ArmEnv, N_ACTIONS, N_JOINTS, OBS_DIM, WIRING, TRIAL_STEPS

run_dir, key_path, name, out_dir = sys.argv[1:5]
torch.set_num_threads(1)
ck = torch.load(key_path, weights_only=False)
keynet = m.SituationKey(); keynet.load_state_dict(ck['net']); keynet.protos = ck['protos']; keynet.eval()
slice_idx, slice_on = m.build_action_slices(keynet)
model = m.A2C_EPN(keynet, slice_idx, slice_on, mem_size=m.config['dict-len'])
model.load_state_dict(torch.load(f'{run_dir}/model.pt', weights_only=False)['model'])
model.eval()

def pack(t):
    a = np.ascontiguousarray(np.asarray(t, np.float32))
    return {'shape': list(a.shape), 'b64': base64.b64encode(a.tobytes()).decode()}

tensors = {k: pack(v.detach().numpy()) for k, v in model.state_dict().items() if k not in ('slice_idx', 'slice_on')}
tensors['protos'] = pack(keynet.protos.numpy())
tensors['slice_on'] = pack(slice_on)
evaluation = json.load(open(f'{run_dir}/evaluation.json'))
agent = {'name': name, 'wiring': WIRING, 'joints': N_JOINTS, 'actions': N_ACTIONS, 'obs': OBS_DIM, 'memory': model.mem_size,
         'slice_idx': slice_idx.tolist(), 'trained_episodes': evaluation['episodes'],
         'held_out_targets': round(evaluation['held_out_targets_reached'], 2), 'tensors': tensors}
with open(f'{out_dir}/{name}.js', 'w') as f:
    f.write('(typeof window !== "undefined" ? window : globalThis).ARM_AGENTS = (typeof window !== "undefined" ? window : globalThis).ARM_AGENTS || {};\n')
    f.write(f'(typeof window !== "undefined" ? window : globalThis).ARM_AGENTS[{json.dumps(name)}] = {json.dumps(agent)};\n')

# trace: one arm, the model's own sampled actions
make_row = m.make_row_fn(slice_idx, slice_on)
env = ArmEnv(1, 4242); state = env.reset()
memory = m.Episodic_Memory(1, model.mem_size, model.mem_dim)
T = lambda x: torch.as_tensor(np.asarray(x)).float()
torch.manual_seed(7); steps = []
with torch.no_grad():
    for _ in range(60):
        trial_t = env.trial_t.copy()
        codes, keys = model.codes_and_keys(state)
        s_in = np.concatenate([state, (trial_t / TRIAL_STEPS)[:, None]], 1)
        _, action, _, _ = model(T(s_in), torch.as_tensor(codes), T(memory.memory), T(memory.generate_mask()),
                                T(keys), T(memory.key_memory), T(memory.key_present))
        a = action.numpy()
        nxt, r, done, new_trial, settled, reached = env.step(a)
        steps.append({'obs': state[0].tolist(), 'trial_t': float(trial_t[0]), 'action': int(a[0]), 'settled': settled[0].tolist(),
                      'reward': float(r[0]), 'logits': model._logits[0].tolist(), 'value': float(model._value[0, 0]),
                      'codes': codes[0].tolist()})
        memory.push(make_row(a, state, settled, r, trial_t), keys[np.arange(1), a])
        state = nxt
json.dump({'name': name, 'steps': steps}, open(f'{out_dir}/{name}.trace.json', 'w'))
print(name, 'exported:', sum(np.prod(t['shape']) for t in tensors.values()), 'numbers, slice width', slice_idx.shape[1])
