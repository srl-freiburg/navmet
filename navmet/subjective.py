r"""

Module doc
"""

from __future__ import division

import itertools
import operator

import numpy as np

from .utils import edist
from .utils import adist
from .utils import distance_to_segment


def personal_disturbance(trajectory, persons, region_type='uniform',
                         regions=[0.45, 1.2, 3.6]):
    """ Personal intrusion counts by a robot onto persons

    Count the number of intrusions into various regions around
    a focal agent.

    Parameters
    -------------
    trajectory : array-like
        Trace of robot poses
    persons : dict
        Dictionary of poses of persons in the environment indexed by id
    region_type : str, optional (default: 'uniform')
        Type of region around a person to consider, options include:
        [uniform, anisotropic]
    regions : list of float, optional (default: [0.45, 1.2, 3.6])
        Radii of regions around the persons to consider if the robot
        passes through, thereby causing intrusion, defaults to Proxemics
        distances (intimate, personal, social). The shape of the regions
        is determined by `region_type` parameter

    Returns
    --------
    ic, pc, sc, : int
        Intimate, Personal and Social region intrusion counts

    """
    # TODO - used a named region list?
    # - better handling of regions

    # check that the regions list is strictly monotonically increasing
    monotone = all(itertools.starmap(operator.le, zip(regions, regions[1:])))
    assert monotone is True,\
        "Regions list must be monotonically increasing"

    trajectory = np.asarray(trajectory)

    ic, pc, sc = 0, 0, 0

    for waypoint in trajectory:
        for idx, person in persons.items():
            if region_type == 'uniform':
                distance = edist(waypoint, person)
            elif region_type == 'anisotropic':
                distance = adist(person, waypoint)
            else:
                raise ValueError('Invalid `region_type`')

            # - cascade check for inclusive lazy count
            if distance < regions[2]:
                sc += 1

                if distance < regions[1]:
                    pc += 1

                    if distance < regions[0]:
                        ic += 1

    return ic, pc, sc


def relation_disturbance(trajectory, relations):
    """ Compute the total disturbance caused on an relation zone

    """
    phi = 0.0
    for waypoint in trajectory:
        for relation in relations:
            dist = distance_to_segment(waypoint, relation)
            if dist < 0.4:
                phi += 1.0

    return phi
