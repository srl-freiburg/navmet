from __future__ import division

import itertools
import operator

import numpy as np
from .angle_utils import normalize
from .general_utils import edist, action_disturbance, distance_segment, gaussianx


def count_uniform_intrusions(focal_agent, other_agents, regions=[0.45, 1.2, 3.6], exclusive=True):
    """
    Count the number of intrusions into various uniform regions around
    a focal agent.

    Parameters
    -------------
    focal_agent : numpy array
        The focal_agent state as a 1D array in format [x, y, vx, vy, ...]
    other_agents : list of numpy arrays
        The other agents as a list of 1D arrays in format [[x, y, vx, vy, ...], ...]
        or a numpy array of shape [agent_features x n_agents]
    regions : list of float, optional (default: [0.45, 1.2, 3.6])
        Radii of regions of the uniform region around the focal_agent to consider,
        defaults to Proxemics distances (intimate, personal, social)
    exclusive: bool
        Flag for whether to coint intrusions in a mutually exclusive sense of cumulative

    Returns
    --------
    ic : int
        Intimate region intrusion counts
    pc : int
        Personal region intrusion counts
    sc : int
        Social region intrusion counts
    """

    # check that the regions list is strictly monotonically increasing
    assert all(itertools.starmap(operator.le, zip(regions, regions[1:]))) is True,\
        "Regions list must be monotonically increasing"

    ic, pc, sc = 0, 0, 0    # TODO - allow more ranges
    if isinstance(other_agents, list):
        other_agents = np.array(other_agents)

    if exclusive:
        for agent in other_agents:
            isc = inside_uniform_region(focal_agent, agent, radius=regions[2])
            if not isc:
                continue

            ipc = inside_uniform_region(focal_agent, agent, radius=regions[1])
            if not ipc:
                sc += 1
                continue

            iic = inside_uniform_region(focal_agent, agent, radius=regions[0])
            if iic:
                ic += 1
            elif not iic and ipc:
                pc += 1
    else:
        for agent in other_agents:
            isc = inside_uniform_region(focal_agent, agent, radius=regions[2])
            if isc:
                sc += 1

            ipc = inside_uniform_region(focal_agent, agent, radius=regions[1])
            if ipc:
                pc += 1

            iic = inside_uniform_region(focal_agent, agent, radius=regions[0])
            if iic:
                ic += 1

    return ic, pc, sc


def count_anisotropic_intrusions(focal_agent, other_agents, aks, exclusive=True):
    """
    Count the number of intrusions into various uniform regions around
    a focal agent.

    Parameters
    -------------
    focal_agent : numpy array
        The focal_agent state as a 1D array in format [x, y, vx, vy, ...]
    other_agents : list of numpy arrays
        The other agents as a list of 1D arrays in format [[x, y, vx, vy, ...], ...]
        or a numpy array of shape [agent_features x n_agents]
    aks : list of float
        ak parameters of the anisotropic region around the focal_agent to consider,

    Returns
    --------
    ic : int
        Intimate region intrusion counts
    pc : int
        Personal region intrusion counts
    sc : int
        Social region intrusion counts
    """

    # check that the regions list is strictly monotonically increasing
    assert all(itertools.starmap(operator.le, zip(aks, aks[1:]))) is True,\
        "aks list must be monotonically increasing"

    ic, pc, sc = 0, 0, 0    # TODO - allow more ranges
    if isinstance(other_agents, list):
        other_agents = np.array(other_agents)

    if exclusive:
        for agent in other_agents:
            isc = inside_anisotropic_region(focal_agent, agent, ak=aks[2])
            if not isc:
                continue

            ipc = inside_anisotropic_region(focal_agent, agent, ak=aks[1])
            if not ipc:
                sc += 1
                continue

            iic = inside_anisotropic_region(focal_agent, agent, ak=aks[0])
            if iic:
                ic += 1
            elif not iic and ipc:
                pc += 1
    else:
        for agent in other_agents:
            isc = inside_anisotropic_region(focal_agent, agent, ak=aks[2])
            if isc:
                sc += 1

            ipc = inside_anisotropic_region(focal_agent, agent, ak=aks[1])
            if ipc:
                pc += 1

            iic = inside_anisotropic_region(focal_agent, agent, ak=aks[0])
            if iic:
                ic += 1

    return ic, pc, sc


def social_relation_disturbance(trajectory, relations):
    """
    """
    srd = 0.0
    for i in xrange(trajectory.shape[0]):
        here = trajectory[i, 0:2]

        for e in relations:
            xaction = here
            sdist, inside = distance_segment(xaction, e[:2], e[2:])
            if inside:
                srd += gaussianx(sdist, mu=0.0, sigma=0.4)
            else:
                ed = min(edist(xaction, e[:2]), edist(xaction, e[2:]))
                srd += gaussianx(ed, mu=0.0, sigma=0.4)

    return srd


def inside_uniform_region(focal_agent, other_agent, radius):
    """
    Check if an agent is inside a given uniform range of another agent as a measure
    of intrusion into the space

    Parameters
    -------------
    focal_agent : numpy array
        The focal_agent state as a 1D array in format [x, y, vx, vy, ...]
    other_agent : numpy array
        The other agent state as a 1D array in format [x, y, vx, vy, ...]
    radius : float
        Radius of the uniform region around the focal_agent to consider

    Returns
    ------------
    inside : bool
        Flag for intrusion presence

    """

    los_distance = edist(focal_agent, other_agent)
    if los_distance <= radius and los_distance > 1e-24:
        return True
    else:
        return False


def inside_anisotropic_region(focal_agent, other_agent, ak=1., bk=1., lmbda=0.4, rij=0.4):
    """
    Check if an agent is inside a given Anisotropic range of another agent as a measure
    of intrusion into the space. Anisotropic regions are circles whose shapes are parametrizable
    as shown;

    .. math::
        a \cdot b \exp{\left(\\frac{r_{ij} - d_{ij}}{b}\\right)} \mathbf{n}_{ij} \left(\lambda + (1 - \lambda) \\frac{1 + \cos(\\varphi_{ij})}{2}\\right)

    Parameters
    -------------
    focal_agent : numpy array
        The focal_agent state as a 1D array in format [x, y, vx, vy, ...]
    other_agent : numpy array
        The other agent state as a 1D array in format [x, y, vx, vy, ...]
    ak : float, optional (default: 1.0)
        Parameter of anisotropic region, size scaling
    bk : float, optional (default: 1.0)
        Parameter of anisotropic region, size scaling
    lmbda : float, optional (default: 0.4)
        Parameter of anisotropic region, controls the shape 'circleness'
    rij : float, optional (default: 0.4)
        Parameter of anisotropic region, sum of agent radii in metres

    Returns
    ------------
    inside : bool
        Flag for intrusion presence

    """

    # euclidean distance between the agents
    dij = edist(focal_agent, other_agent)

    look_vector = np.array([-focal_agent[2], -focal_agent[3]])
    len_lv = np.linalg.norm(look_vector)

    if len_lv == 0.0:
        ei = look_vector
    else:
        ei = look_vector / len_lv
    phi = normalize(np.arctan2(other_agent[1] - focal_agent[1], other_agent[0] - focal_agent[0]))
    nij = np.array([np.cos(phi), np.sin(phi)])  # normalized vector pointing from j to i

    alpha = ak * np.exp((rij - dij) / bk) * nij
    beta_ = np.tile(np.ones(shape=(1, 2)) * lmbda + ((1 - lmbda)
                    * (np.ones(shape=(1, 2)) - (np.dot(nij.T, ei)).T) / 2.), [1, 1])
    c1 = np.multiply(alpha, beta_)
    curve = c1.T
    dc = np.sqrt(curve[0] ** 2 + curve[1] ** 2)

    if dij <= dc:
        return True
    else:
        return False
