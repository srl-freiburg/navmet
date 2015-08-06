from __future__ import division

import itertools
import numpy as np

from .angle_utils import subangles, normalize


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
    for i, j in itertools.izip(range(trajectory.shape[0]),
                               range(1, trajectory.shape[0])):
        if not timestamped:
            current, nextstep = trajectory[i, 0:2], trajectory[j, 0:2]
        else:
            current, nextstep = trajectory[i, 1:3], trajectory[j, 1:3]
        path_length += np.linalg.norm(current - nextstep)

    return path_length


def cumulative_heading_changes(trajectory, timestamped=False,
                               degrees=False, xytheta=False):
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

    if xytheta:
        theta_old = ipoint[2]
    else:
        if timestamped:
            theta_old = normalize(np.arctan2(ipoint[4], ipoint[3]))
        else:
            theta_old = normalize(np.arctan2(ipoint[3], ipoint[2]))

    theta_acc = 0
    for i, j in itertools.izip(range(trajectory.shape[0]),
                               range(1, trajectory.shape[0])):
        if not timestamped:
            current, nextstep = trajectory[i, 0:2], trajectory[j, 0:2]
        else:
            current, nextstep = trajectory[i, 1:3], trajectory[j, 1:3]

        x1, y1 = current[0], current[1]
        x2, y2 = nextstep[0], nextstep[1]
        dx, dy = (x2 - x1), (y2 - y1)

        if abs(dx) > 1e-10 or abs(dy) > 1e-10:
            theta_i = normalize(np.arctan2(dy, dx))
            delta_theta = abs(subangles(theta_i, theta_old))
            theta_acc = theta_acc + delta_theta
            theta_old = theta_i

    if degrees is False:
        return theta_acc
    else:
        return np.degrees(theta_acc)


def edge_crossing(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if the edges [[x1,y1][x2,y2]] and [[x3,y3][x4,y4]] cross themselves

    Parameters
    -------------
    x1,y1,x2,y2,x3,y3,x4,y4 : float
        coordonates of the points defining the edges

    Returns
    -----------
    cross : int
        0 (False) or 1 (True)
    """
    if x2 == x1:
        if x3 == x4:
            if x1 != x3:
                return 0
            elif max(y3, y4) < min(y1, y2) or max(y1, y2) < min(y3, y4):
                return 0
            else:
                return 1
        else:
            a2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - (a2 * x3)
            if a2 == 0:
                if min(y1, y2) > b2 or max(y1, y2) < b2:
                    return 0
                elif x2 <= max(x3, x4) and x2 >= min(x3, x4):
                    return 1
                else:
                    return 0
            elif a2 * x1 + b2 <= min(max(y3, y4), max(y1, y2)) and \
                    a2 * x1 + b2 >= max(min(y3, y4), min(y1, y2)):
                return 1
            else:
                return 0
    elif x3 == x4:
        if x1 == x2:
            if x1 != x3:
                return 0
            elif max(y3, y4) < min(y1, y2) or max(y1, y2) < min(y3, y4):
                return 0
            else:
                return 1
        else:
            a1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - (a1 * x1)
            if a1 == 0:
                if min(y3, y4) > b1 or max(y3, y4) < b1:
                    return 0
                elif x3 <= max(x1, x2) and x3 >= min(x1, x2):
                    return 1
                else:
                    return 0
            elif a1 * x3 + b1 <= min(max(y1, y2), max(y3, y4)) and \
                    a1 * x3 + b1 >= max(min(y1, y2), min(y3, y4)):
                return 1
            else:
                return 0
    else:
        a1 = (y2 - y1) / (x2 - x1)
        a2 = (y4 - y3) / (x4 - x3)
        if a1 == a2:
            return 0
        else:
            b2 = y3 - (a2 * x3)
            b1 = y1 - (a1 * x1)
            xcommun = (b2 - b1) / (a1 - a2)
            if xcommun >= max(min(x1, x2), min(x3, x4)) and \
                    xcommun <= min(max(x1, x2), max(x3, x4)):
                return 1
            else:
                return 0
