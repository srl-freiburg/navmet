from __future__ import division

import itertools
import numpy as np

from .angle_utils import subangles


def path_length(trajectory, timestamped=False):
    """
    Compute the length of a path travelled by an agent

    Parameters
    -----------
    trajectory : numpy array
        Trajectory representing the path travelled as a `numpy` array of
        shape [frame_size x n_waypoints]. Frame encodes information at
        every time step [time, x, y, vx, vy, ...]
    timestamped : bool, optional (default: False)
        Flag fot trajectories which contain time information

    Returns
    ---------
    path_length : float
        Length of the path based on Euclidean distance metric
    """

    assert trajectory.ndim == 2, "Trajectory must be a two dimensional array"

    path_length = 0.0
    for i, j in itertools.izip(xrange(trajectory.shape[0]), xrange(1, trajectory.shape[0])):
        if not timestamped:
            current, nextstep = trajectory[i, 0:2], trajectory[j, 0:2]
        else:
            current, nextstep = trajectory[i, 1:3], trajectory[j, 1:3]
        path_length += np.linalg.norm(current - nextstep)

    return path_length


def cumulative_heading_changes(trajectory, timestamped=False, degrees=False):
    """
    Count the cumulative heading changes of in the trajectory
    measured by angles between succesive waypoints. Gives
    a simple way to check on smoothness of path and energy

    Parameters
    -----------
    trajectory : numpy array
        Trajectory representing the path travelled as a `numpy` array of
        shape [frame_size x n_waypoints]. Frame encodes information at
        every time step [time, x, y, vx, vy, ...]
    timestamped : bool, optional (default: False)
        Flag fot trajectories which contain time information
    degrees : bool, optional (default: False)
        Flag to return cumulative heading changes in degrees

    Returns
    ---------
    theta_acc : float
        Cumulative heading changes in angles
    """

    assert trajectory.ndim == 2, "Trajectory must be a two dimensional array"

    ipoint = trajectory[0, :]
    if timestamped:
        theta_old = math.atan2(ipoint[4], ipoint[3])
    else:
        theta_old = math.atan2(ipoint[3], ipoint[2])

    theta_acc = 0
    for i, j in itertools.izip(xrange(trajectory.shape[0]), xrange(1, trajectory.shape[0])):
        if not timestamped:
            current, nextstep = trajectory[i, 0:2], trajectory[j, 0:2]
        else:
            current, nextstep = trajectory[i, 1:3], trajectory[j, 1:3]

        x1, y1 = current[0], current[1]
        x2, y2 = nextstep[0], nextstep[1]
        dx, dy = (x2 - x1), (y2 - y1)

        if abs(dx) > 1e-10 or abs(dy) > 1e-10:
            theta_i = np.arctan2(dy, dx)
            delta_theta = abs(subangles(theta_i, theta_old))
            theta_acc = theta_acc + delta_theta
            theta_old = theta_i

    if degrees is False:
        return theta_acc
    else:
        return np.degrees(theta_acc)
