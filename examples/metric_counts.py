

import os
import sys
# local devel testing without installing the package
sys.path = [os.path.abspath(os.path.join(os.getcwd(),'..')), ] + sys.path


import navmet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import seaborn as sns

## Subjective metrics
# load some sample agents
agents = np.loadtxt('sample_agents.txt')

# pick a focal agent
focal_agent = agents[0, :]
other_agents = agents[1:, :]

ic, pc, sc = navmet.count_uniform_intrusions(focal_agent, other_agents)
print('Instrusion Counts: Intimate = {}, Personal = {}, Social = {}'.format(ic, pc, sc))


## Objective metrics
traj = np.loadtxt('sample_traj.txt')
pl = navmet.path_length(trajectory=traj)
chc = navmet.cumulative_heading_changes(trajectory=traj)
print('Objective Metrics: Path length = {}, Cum. heading changes = {}'.format(pl, chc))


# plot their configuration
plt.figure(figsize=(8, 6))
plt.plot(agents[:, 0], agents[:, 1], ls='', marker='8', markersize=15, label='Sample agents')
plt.plot(traj[:, 0], traj[:, 1], ls='-', lw=1.5, label='Sample trajectory')
plt.legend(loc='best', numpoints=1)

# zooming in to the context
plt.figure(figsize=(8, 8))
ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
ax.plot(0.0, 0.0, marker='8', markersize=20, color='r', linestyle='', label='Focal Agent')
ax.add_artist(Circle((0.0, 0.0), radius=3.6, color='r', fill=False, hatch='.', alpha=0.3, lw=2.))

for n in agents:
    heading = np.degrees(np.arctan2(n[3], n[2]))
    ax.add_artist(Ellipse((n[0] - focal_agent[0], n[1] - focal_agent[1]), width=0.3, height=0.8, angle=heading, color='b', fill=False, lw=1.5))
    ax.add_artist(Circle((n[0] - focal_agent[0], n[1] - focal_agent[1]), radius=0.2, color='b'))
    ax.arrow(n[0] - focal_agent[0], n[1] - focal_agent[1], n[2], n[3], fc='b', ec='b', head_width=0.1, head_length=0.1)

plt.axis('equal')
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.legend(loc='upper right', numpoints=1, markerscale=.5)

plt.show()
