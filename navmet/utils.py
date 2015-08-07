from __future__ import division

import numpy as np


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
    if theta < np.inf:
        while theta >= start + 2 * np.pi:
            theta -= 2 * np.pi
        while theta < start:
            theta += 2 * np.pi
        return theta
    else:
        return np.inf


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
    if alpha < np.inf and beta < np.inf:
        alpha = normalize(alpha, start=0)
        beta = normalize(beta, start=0)

        delta = alpha - beta
        if alpha > beta:
            while delta > np.pi:
                delta -= 2 * np.pi
        elif beta > alpha:
            while delta < -np.pi:
                delta += 2 * np.pi
    else:
        delta = np.inf

    return delta


def edist(v1, v2):
    """ Euclidean distance between the two poses

    Parameters
    -----------
    v1, v2 : array-like
        vector of poses

    Returns
    -----------
    dist : float
        distance between v1 and v2
    """
    return np.hypot((v1[0] - v2[0]), (v1[1] - v2[1]))


def distance_segment(point, line_start, line_end):
    """ Distance from a point to a line segment

    Distance between a point `point` and a line segment defined by two points
    `line_start` and `line_end`. Additionally gives information of whether
    the point lies within the perpendicular lines from either end of the
    line segment.

    """
    xa = line_start[0]
    ya = line_start[1]
    xb = line_end[0]
    yb = line_end[1]
    xp = point[0]
    yp = point[1]

    # x-coordinates
    A = xb-xa
    B = yb-ya
    C = yp*B+xp*A
    a = 2*((B*B)+(A*A))
    b = -4*A*C+(2*yp+ya+yb)*A*B-(2*xp+xa+xb)*(B*B)
    c = 2*(C*C)-(2*yp+ya+yb)*C*B+(yp*(ya+yb)+xp*(xa+xb))*(B*B)
    x1 = (-b + np.sqrt((b*b)-4*a*c))/(2*a)
    x2 = (-b - np.sqrt((b*b)-4*a*c))/(2*a)

    # y-coordinates
    A = yb-ya
    B = xb-xa
    C = xp*B+yp*A
    a = 2*((B*B)+(A*A))
    b = -4*A*C+(2*xp+xa+xb)*A*B-(2*yp+ya+yb)*(B*B)
    c = 2*(C*C)-(2*xp+xa+xb)*C*B+(xp*(xa+xb)+yp*(ya+yb))*(B*B)
    y1 = (-b + np.sqrt((b*b)-4*a*c))/(2*a)
    y2 = (-b - np.sqrt((b*b)-4*a*c))/(2*a)

    # Put point candidates together
    xfm1 = np.array([x1, y1])
    xfm2 = np.array([x2, y2])
    xfm3 = np.array([x1, y2])
    xfm4 = np.array([x2, y1])

    dvec = list()
    dvec.append(edist(xfm1, point))
    dvec.append(edist(xfm2, point))
    dvec.append(edist(xfm3, point))
    dvec.append(edist(xfm4, point))

    dmax = -1
    imax = -1
    for i in xrange(4):
        if dvec[i] > dmax:
            dmax = dvec[i]
            imax = i

    xf = xfm1
    if imax == 0:
        xf = xfm1
    elif imax == 1:
        xf = xfm2
    elif imax == 2:
        xf = xfm3
    elif imax == 3:
        xf = xfm4

    line_start_xf = np.array([line_start[0]-xf[0], line_start[1]-xf[1]])
    line_end_xf = np.array([line_end[0]-xf[0], line_end[1]-xf[1]])
    dotp = np.dot(line_end_xf, line_start_xf)
    inside = np.sign(dotp)

    return dmax, inside


def gaussianx(x, mu, sigma=0.2):
    """
    Evaluate a Gaussian at a point
    """
    fg = (1.0 / (sigma * np.sqrt(2*np.pi))) *\
        np.exp(-(x - mu)*(x - mu) / (2.0 * sigma * sigma))
    return fg / 1.0
