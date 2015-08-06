from __future__ import division

import math


__all__ = ['normalize', 'addangles', 'subangles']


def normalize(theta, start=0):
    """
    Normalize an angle to be in the range :math:`[0, 2\pi]`

    Parameters
    -----------
    theta : float
        input angle to normalize

    start: float
        input start angle (optional, default: 0.0)

    Returns
    --------
    res : float
        normalized angle or :math:`\infty`

    """
    if theta < float("inf"):
        while theta >= start + 2 * math.pi:
            theta -= 2 * math.pi
        while theta < start:
            theta += 2 * math.pi
        return theta
    else:
        return float("inf")


def addangles(alpha, beta):
    """
    Add two angles

    Parameters
    ----------
    alpha : float
        Augend (in radians)
    beta : float
        Addend (in radians)

    Returns
    -------
    sum : float
        Sum (in radians, normalized to [0, 2pi])
    """
    return normalize(alpha + beta, start=0)


def subangles(alpha, beta):
    """
    Substract one angle from another

    Parameters
    ----------
    alpha : float
        Minuend (in radians)
    beta : float
        Subtraend (in radians)

    Returns
    -------
    delta : float
        Difference (in radians, normalized to [0, 2pi])
    """
    delta = 0
    if alpha < float("inf") and beta < float("inf"):
        alpha = normalize(alpha, start=0)
        beta = normalize(beta, start=0)

        delta = alpha - beta
        if alpha > beta:
            while delta > math.pi:
                delta -= 2 * math.pi
        elif beta > alpha:
            while delta < -math.pi:
                delta += 2 * math.pi
    else:
        delta = float("inf")

    return delta
