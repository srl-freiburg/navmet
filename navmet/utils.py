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


def anisotropic_distance(focal_agent, other_agent,
                         phi_ij=None, ak=2.48, bk=1.0,
                         lambda_=0.4, rij=0.9):
    """
    Anisotropic distance based on the Social Force Model (SFM)
    model of pedestrian dynamics.
    """
    ei = np.array([-focal_agent[2], -focal_agent[3]])
    length_ei = np.linalg.norm(ei)
    if length_ei > 1e-24:
        ei = ei / length_ei

    if phi_ij is None:
        phi = np.arctan2(other_agent[1] - focal_agent[1],
                         other_agent[0] - focal_agent[0])
    else:
        phi = phi_ij

    dij = edist(focal_agent, other_agent)
    nij = np.array([np.cos(phi), np.sin(phi)])
    ns = 2
    alpha = ak * np.exp((rij - dij) / bk) * nij
    beta_ = np.tile(np.ones(shape=(1, ns)) * lambda_ + ((1 - lambda_)
                    * (np.ones(shape=(1, ns)) - (np.dot(nij.T, ei)).T) / 2.),
                    [1, 1])
    curve = np.multiply(alpha, beta_).T
    dc = np.hypot(curve[0], curve[1])
    return dc


def distance_to_segment(x, xs, xe):
    xa = xs[0]
    ya = xs[1]
    xb = xe[0]
    yb = xe[1]
    xp = x[0]
    yp = x[1]

    # x-coordinates
    A = xb-xa
    B = yb-ya
    C = yp*B+xp*A
    a = 2*((B*B)+(A*A))
    b = -4*A*C+(2*yp+ya+yb)*A*B-(2*xp+xa+xb)*(B*B)
    c = 2*(C*C)-(2*yp+ya+yb)*C*B+(yp*(ya+yb)+xp*(xa+xb))*(B*B)
    if b*b < 4*a*c:
        return None, False
    x1 = (-b + np.sqrt((b*b)-4*a*c))/(2*a)
    x2 = (-b - np.sqrt((b*b)-4*a*c))/(2*a)

    # y-coordinates
    A = yb-ya
    B = xb-xa
    C = xp*B+yp*A
    a = 2*((B*B)+(A*A))
    b = -4*A*C+(2*xp+xa+xb)*A*B-(2*yp+ya+yb)*(B*B)
    c = 2*(C*C)-(2*xp+xa+xb)*C*B+(xp*(xa+xb)+yp*(ya+yb))*(B*B)
    if b*b < 4*a*c:
        return None, False
    y1 = (-b + np.sqrt((b*b)-4*a*c))/(2*a)
    y2 = (-b - np.sqrt((b*b)-4*a*c))/(2*a)

    # Put point candidates together
    xfm1 = [x1, y1]
    xfm2 = [x2, y2]
    xfm3 = [x1, y2]
    xfm4 = [x2, y1]

    dvec = list()
    dvec.append(edist(xfm1, x))
    dvec.append(edist(xfm2, x))
    dvec.append(edist(xfm3, x))
    dvec.append(edist(xfm4, x))

    dmax = -1.0
    imax = -1
    for i in range(4):
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

    xs_xf = [xs[0]-xf[0], xs[1]-xf[1]]
    xe_xf = [xe[0]-xf[0], xe[1]-xf[1]]
    dotp = (xs_xf[0] * xe_xf[0]) + (xs_xf[1] * xe_xf[1])

    inside = False
    if dotp <= 0.0:
        inside = True

    return dmax, inside


def gaussianx(x, mu, sigma=0.2):
    """
    Evaluate a Gaussian at a point
    """
    fg = (1.0 / (sigma * np.sqrt(2*np.pi))) *\
        np.exp(-(x - mu)*(x - mu) / (2.0 * sigma * sigma))
    return fg / 1.0
