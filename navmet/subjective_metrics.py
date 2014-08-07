from __future__ import division

# intrusion counts in personal spaces
# - uniform
# - anisotropic
# - elliptic -> later



def inside_uniform_region(focal_agent, other_agent, radius):
    """
    Check if an agent is inside a given uniform range of another agent as a measure
    of intrusion into the space

    Parameters
    -------------
    focal_agent : numpy array
        The focal_agent state as a 1D array in format [x, y, vx, vy, ...]
    other_agent : list of AgentState
        The other agent state as a 1D array in format [x, y, vx, vy, ...]
    radius : float
        Radius of the uniform region around the focal_agent to consider

    Returns
    ------------
    inside : bool
        Flag for intrusion presence

    """

    los_distance = np.linalg.norm(focal_agent[0:2] - other_agent[0:2])
    if los_distance <= radius and los_distance > 1e-12:
        return True
    else:
        return False


def inside_anisotropic_region(focal_agent, other_agent, ak=1., bk=1., lambda_=0.4, rij=0.4):
    """
    Check if an agent is inside a given Anisotropic range of another agent as a measure
    of intrusion into the space. Anisotropic regions are circles whose shapes are parametrizable
    as shown;

    ..math::


    Parameters
    -------------
    focal_agent : numpy array
        The focal_agent state as a 1D array in format [x, y, vx, vy, ...]
    other_agent : list of AgentState
        The other agent state as a 1D array in format [x, y, vx, vy, ...]
    ak : float, optional (default: 1.0)
        Parameter of anisotropic region, size scaling
    bk : float, optional (default: 1.0)
        Parameter of anisotropic region, size scaling
    lambda_ : float, optional (default: 0.4)
        Parameter of anisotropic region, controls the shape 'circleness'
    rij : float, optional (default: 0.4)
        Parameter of anisotropic region, sum of agent radii in metres

    Returns
    ------------
    inside : bool
        Flag for intrusion presence

    """

    # euclidean distance between the agents
    dij = np.linalg.norm(focal_agent[0:2] - other_agent[0:2])

    look_vector = np.array([-focal_agent.vx, -focal_agent.vy])
    len_lv = np.linalg.norm(look_vector)

    if len_lv == 0.0:
        ei = look_vector
    else:
        ei = look_vector / len_lv
    phi = normalize(math.atan2(other_agent.y - focal_agent.y, other_agent.x - focal_agent.x))
    nij = np.array([np.cos(phi), np.sin(phi)])  # normalized vector pointing from j to i

    alpha = ak * np.exp((rij - dij) / bk) * nij
    beta_ = np.tile(np.ones(shape=(1, 2)) * lambda_ + ((1 - lambda_)
                    * (np.ones(shape=(1, 2)) - (np.dot(nij.T, ei)).T) / 2.), [1, 1])
    c1 = np.multiply(alpha, beta_)
    curve = c1.T
    dc = np.sqrt(curve[0] ** 2 + curve[1] ** 2)

    if dij <= dc:
        return True
    else:
        return False
