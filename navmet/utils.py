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


def adist(focal_agent, other_agent, ak=2.48, bk=1.0, lambda_=0.4, rij=0.9):
    """ Anisotropic distance between two oriented poses

    Anisotropic distance based on the Social Force Model (SFM) [TODO - cite]
    model of pedestrian dynamics.

    .. math::
        a \cdot b \exp{\left(\\frac{r_{ij} - d_{ij}}{b}\\right)}
        \mathbf{n}_{ij} \left(\lambda + (1 - \lambda) \\frac{1 +
                              \cos(\\varphi_{ij})}{2}\\right)

    Parameters
    -----------
    focal_agent, other_agent : array-like
        Vector of poses (including orientation information as vx, vy)
    ak, bk, lambda_, rij : float
        Parameters of the anisotropic model


    Returns
    ----------
    dist : float
        Distance between the two poses

    """
    ei = np.array([-focal_agent[2], -focal_agent[3]])
    length_ei = np.linalg.norm(ei)
    if length_ei > 1e-24:
        ei = ei / length_ei

    phi = np.arctan2(other_agent[1] - focal_agent[1],
                     other_agent[0] - focal_agent[0])

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


def distance_to_segment(x, (xs, xe)):
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


def extract_relations(persons, groups):
    """" Extract relation links from grouping information

    Given poses of persons and grouping information in form of person ids per
    group, this method extracts line segments representing the relation
    links between the persons.

    Parameters
    ----------
    persons : dict
        Dictionary of person poses indexed by id
    groups : array-like
        2D array with each row containing ids of a pairwise grouping. For
        groups with more than 2 persons, multiple rows are used for every to
        represent every pairing possible

    Returns
    --------
    elines : array-like
        An a array of line segments, each represented by a tuple of start and
        end points
    """
    elines = []
    for [i, j] in groups:
        line = ((persons[i][0], persons[i][1]), (persons[j][0], persons[j][1]))
        elines.append(line)

    return elines


def dtw(x, y, dist=lambda x, y: np.linalgnorm(x - y, ord=1)):
    """ Computes the dtw between two signals.

    Adapted from: https://github.com/pierre-rouanet/dtw/blob/master/dtw.py
    """
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    y = np.array(y)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y)

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] = dist(x[i], y[j])

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += min(D[i, j], D[i, j+1], D[i+1, j])

    D = D[1:, 1:]
    dist = D[-1, -1] / sum(D.shape)

    return dist, D, _track_back(D)


def _track_back(D):
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while i > 0 and j > 0:
        tb = np.argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))

        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        elif tb == 2:
            j -= 1

        p.insert(0, i)
        q.insert(0, j)

    p.insert(0, 0)
    q.insert(0, 0)

    return (np.array(p), np.array(q))
