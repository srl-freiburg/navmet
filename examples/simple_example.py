

import navmet

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
plt.style.use('fivethirtyeight')


def simple_example():
    # -- Objective metrics
    traj = np.loadtxt('sample_traj.txt')
    agents = np.loadtxt('sample_agents.txt')
    # robot = agents[0, :]
    persons = agents[1:, :]

    # --- Objective
    pl = navmet.path_length(trajectory=traj)
    chc = navmet.chc(trajectory=traj)
    print('Objective Metrics: Path length = {}, CHC = {}'.format(pl, chc))

    # plot their configuration
    plt.figure(figsize=(10, 10))
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    for n in agents:
        theta = np.degrees(np.arctan2(n[3], n[2]))
        ax.add_artist(Ellipse((n[0], n[1]), width=0.3, height=0.8, angle=theta,
                      color='b', fill=False, lw=1.5))
        ax.add_artist(Circle((n[0], n[1]), radius=0.2, color='w',
                      ec='b', lw=4))
        ax.arrow(n[0], n[1], 0.5*n[2], 0.5*n[3], fc='b', ec='b',
                 head_width=0.2, head_length=0.2)

    ax.plot(traj[::3, 0], traj[::3, 1], ls='', marker='8', markersize=5,
            color='r', label='Robot trajectory')
    ax.plot(traj[0, 0], traj[0, 1], ls='', marker='8', markersize=15,
            color='k', label='Start')
    ax.plot(traj[-1, 0], traj[-1, 1], ls='', marker='8', markersize=15,
            color='g', label='Goal')
    ax.legend(loc='best', numpoints=1)
    plt.axis('equal')

    # --- Subjective
    ic, pc, sc = navmet.personal_disturbance(traj, persons)
    print('Instrusion Counts: Intimate = {}, Personal = {}, Social = {}'
          .format(ic, pc, sc))

    plt.show()


if __name__ == '__main__':
    simple_example()
