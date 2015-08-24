from __future__ import division

import itertools
import numpy as np

from .utils import subangles, normalize, dtw

SMALL_CHANGE = 1e-10


__all__ = ['path_length', 'chc', 'path_similarity']


def path_length(traj, metric=lambda x, y: np.linalg.norm(x - y, ord=2)):
    """ Compute the length of a robot trajectory

    Compute the sum of distances between waypoints in the robot trajectory
    using specified distance metric. Given a pose :math:`\mathbf{p} = (x, y)`

    .. math::

        l = \sum^{T}_{i=1} \mu( \mathbf{p}_i - \mathbf{p}_{i-1} )

    where :math:`\mu` is a *metric*.

    Parameters
    -----------
    traj : array-like
        Trajectory representing the path travelled as an array. Each frame
        can encode information such as (x, y, theta, speed, ...)
    metric : callable, optional (default: Euclidean distance)
        A metric callable taking two poses and retuning a real value

    Returns
    ---------
    path_length : float
        Length of the path based on given distance metric

    """
    traj = np.asarray(traj)
    assert traj.ndim == 2, "Trajectory must be a two dimensional array"

    path_length = 0.0
    path_length = sum(metric(traj[i, 0:2], traj[j, 0:2])
                      for i, j in itertools.izip(range(traj.shape[0]),
                                                 range(1, traj.shape[0])))

    return path_length


def chc(traj, degrees=False):
    """ Count the cumulative heading changes along a trajectory

    Count the cumulative heading changes of in the trajectory
    measured by angles between succesive waypoints. Gives
    a simple way to check on smoothness of path and energy

    Parameters
    -----------
    traj : array-like
        Trajectory representing the path travelled as an array. Each frame
        can encode information such as (x, y, theta, speed, ...)

    degrees : bool, optional (default: False)
        Flag to return cumulative heading changes in degrees

    Returns
    ---------
    chc : float
        Cumulative heading changes in angles


    Note
    ------
    This method currently does not use orientation information if it is
    available. Instead it recomputes the angles between waypoints in order
    to get the heading changes. This may change in the future

    """
    traj = np.asarray(traj)
    assert traj.ndim == 2, "Trajectory must be a two dimensional array"

    heading_changes = 0.0
    theta_old = 0.0
    for i, j in itertools.izip(range(traj.shape[0]), range(1, traj.shape[0])):
        current, nextstep = traj[i, 0:2], traj[j, 0:2]
        x1, y1 = current[0], current[1]
        x2, y2 = nextstep[0], nextstep[1]
        dx, dy = (x2 - x1), (y2 - y1)

        if abs(dx) > SMALL_CHANGE or abs(dy) > SMALL_CHANGE:
            theta_i = normalize(np.arctan2(dy, dx))
            delta_theta = abs(subangles(theta_i, theta_old))
            heading_changes = heading_changes + delta_theta
            theta_old = theta_i

    if degrees is False:
        return heading_changes

    return np.degrees(heading_changes)


def path_similarity(traj1, traj2, method='dtw'):
    """ Compute the similarity between two paths

    Paths are treated as different time series signals.
    """
    dist, _, _ = dtw(traj1, traj2)

    return dist
