"""Dot plot: targets per episode on damage no agent trained on, with no retraining.
python plot_ood.py results/unseen-damage results/unseen_damage.png"""
import sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID, BLUE, ORANGE, VIOLET = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#7a4fc4'
load = lambda f: json.load(open(f'{D}/{f}.json'))
ours, ours2, refs = load('retrieval-a'), load('retrieval-b'), load('controllers')
rl_same, rl_long, maml = load('recurrent-1000'), load('recurrent-4000'), load('maml')
ROWS = [('trained range', 'Inside the training range'), None,
        ('strength 2.0', 'Every motor 2x stronger than trained'), ('strength 3.0', 'Every motor 3x stronger'),
        ('crosstalk 0.5', 'Each motor also drags a second joint (half)'), ('crosstalk 1.0', 'Each motor also drags a second joint (fully)'),
        ('crosstalk -1.0', 'Each motor pushes a second joint backwards'),
        ('jitter 0.3', 'Jittery motors (30% error per move)'), ('jitter 0.6', 'Jittery motors (60% error per move)'), None,
        ('strength 0.5', 'Every motor at half strength (edge of range)'), ('strength 0.35', 'Every motor at 0.35 strength'),
        ('strength 0.25', 'Every motor at quarter strength'), ('strength 0.15', 'Every motor at 0.15 strength')]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)
y, ticks, labels = 0, [], []
for row in ROWS:
    if row is None:
        y -= 0.6; continue
    key, label = row
    vals = [refs[key]['calibrate'], rl_same[key], rl_long[key], ours[key], ours2[key], maml[key]]
    ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle='round')
    ax.scatter(refs[key]['calibrate'], y, marker='|', s=260, color=INK2, lw=2, zorder=2, label='Hand-written self-calibrating controller')
    ax.scatter(maml[key], y, marker='D', s=42, color=VIOLET, zorder=3, label='MAML, 4,000 training episodes')
    ax.scatter(rl_same[key], y, s=60, facecolor=SURF, edgecolor=ORANGE, lw=2, zorder=3, label='Recurrent agent, 1,000 training episodes')
    ax.scatter(rl_long[key], y, s=60, color=ORANGE, zorder=3, label='Recurrent agent, 4,000 training episodes')
    ax.scatter([ours[key], ours2[key]], [y, y], s=80, color=BLUE, edgecolor=SURF, lw=1, zorder=4, label='Retrieval agent, 1,000 training episodes (two trained copies)')
    ticks.append(y); labels.append(label); y -= 1
ax.set_yticks(ticks); ax.set_yticklabels(labels, color=INK)
ax.set_xlabel('Targets reached per episode (128 unseen arms, no retraining)', color=INK2)
ax.set_xlim(0, None); ax.grid(axis='x', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
h, l = ax.get_legend_handles_labels()
seen = dict(zip(l, h))
order = ['Retrieval agent, 1,000 training episodes (two trained copies)', 'Recurrent agent, 4,000 training episodes', 'Recurrent agent, 1,000 training episodes', 'MAML, 4,000 training episodes', 'Hand-written self-calibrating controller']
leg = fig.legend([seen[k] for k in order], order, loc='upper left', bbox_to_anchor=(0.02, 0.945), ncol=2, frameon=False, fontsize=9.5)
for t in leg.get_texts(): t.set_color(INK)
fig.suptitle('Damage nobody trained on, handled with no retraining', x=0.02, y=0.985, ha='left', color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(OUT, facecolor=SURF)
