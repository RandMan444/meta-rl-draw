"""Draw the comparison chart from the logs in results/:  python python/plot_results.py results/harder_arms.png"""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
SURF, INK, INK2, GRID, BLUE, ORANGE = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834'
PANELS = [('Original arm: 3 motors, 12 actions', 'original-arm', 10.12),          # folder, classical self-calibration score
          ('Miswired board: 3 joints, 12 buttons', 'board3x12', 9.60),
          ('Miswired board: 3 joints, 24 buttons', 'board3x24', 9.20),
          ('Miswired board: 5 joints, 20 buttons', 'board5x20', 7.41)]

def curve(path, k=25):
    f = os.path.join(path, 'episodes.csv')
    if not os.path.exists(f): return None
    y = np.array([float(r['targets_reached']) for r in csv.DictReader(open(f))])
    return (np.arange(k, len(y) + 1), np.convolve(y, np.ones(k) / k, mode='valid')) if len(y) >= k else None

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), dpi=200, facecolor=SURF, sharey=True)
for ax, (title, folder, classical) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    ax.axhline(classical, color=INK2, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.text(1490, classical + 0.2, 'classical self-calibration', color=INK2, fontsize=8.5, ha='right', va='bottom')
    for kind, color, label in (('recurrent', ORANGE, 'Recurrent meta-RL agent (LSTM, A2C)'), ('retrieval', BLUE, 'Nearest-neighbor retrieval agent')):
        first = True
        for seed in (0, 1):
            c = curve(os.path.join(ROOT, folder, f'{kind}-s{seed}'))
            if c is None: continue
            ax.plot(c[0][c[0] <= 1500], c[1][c[0] <= 1500], color=color, lw=2, zorder=3, label=label if first else None, solid_capstyle='round')
            first = False
    ax.set_xlim(0, 1500); ax.set_ylim(0, 11.5); ax.set_xticks([0, 500, 1000, 1500])
    ax.set_title(title, loc='left', color=INK, fontsize=10.5, pad=8); ax.set_xlabel('Training episode', color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
axes[0].set_ylabel('Targets reached per episode', color=INK2)
h, l = axes[0].get_legend_handles_labels()
leg = fig.legend(h[::-1], l[::-1], loc='upper left', bbox_to_anchor=(0.045, 1.0), ncol=2, frameon=False, fontsize=10)
for t in leg.get_texts(): t.set_color(INK)
fig.suptitle('Learning to control a rewired arm: the more hidden wiring, the wider the gap', x=0.985, y=0.985, ha='right', color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(sys.argv[1], facecolor=SURF)
