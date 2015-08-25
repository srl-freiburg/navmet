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


REGION_TYPES = ('anisotropic', 'uniform')


def personal_disturbance(traj, persons, region_type='uniform',
                         regions=[0.45, 1.2, 3.6]):
    """ Personal intrusion counts by a robot onto persons

    Count the number of intrusions into various regions around
    a focal agent.

    Parameters
    -------------
    traj : array-like
        Trace of robot poses
    persons : array-like
        Array of poses of persons in the environment indexed by id
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
    intimate, personal, social : int
        Intimate, Personal and Social region intrusion counts


    Note
    ------
    This methods assumes static scenes in which the poses of persons are
    fixed while the robot moves around. For dynamic cases one can use the
    dynamic version `personal_disturbance_dynamic`

    """
    # TODO - used a named region list?
    # - better handling of regions

    # check that the regions list is strictly monotonically increasing
    monotonic = all(itertools.starmap(operator.le, zip(regions, regions[1:])))
    assert monotonic, "Regions list must be monotonically increasing"

    assert region_type in REGION_TYPES, \
        'Supported region types: {}'.format(REGION_TYPES)

    traj = np.asarray(traj)
    assert traj.ndim == 2, "`traj` must be a two dimensional array"

    persons = np.asarray(persons)
    assert persons.ndim == 2, "`persons` must be a two dimensional array"

    intimate, personal, social = 0, 0, 0
    for waypoint in traj:
        i, p, s = _personal_disturbance(waypoint, persons,
                                        region_type, regions)
        intimate += i
        personal += p
        social += s

    return intimate, personal, social


def personal_disturbance_dynamic(traj, persons, region_type='uniform',
                                 regions=[0.45, 1.2, 3.6]):
    """ Personal intrusion counts by a robot onto persons

    Count the number of intrusions into various regions around
    a focal agent.

    Parameters
    -------------
    traj : array-like
        Trace of robot poses
    persons : array-like
        3D array of poses of persons in the environment indexed by frame
        for every waypoint in the trajectory
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
    intimate, personal, social : int
        Intimate, Personal and Social region intrusion counts


    Note
    ------
    This methods assumes dynamic scenes in which the poses of persons are
    fixed while the robot moves around. For static cases one can use the
    version `personal_disturbance`

    """

    # check that the regions list is strictly monotonically increasing
    monotonic = all(itertools.starmap(operator.le, zip(regions, regions[1:])))
    assert monotonic, "Regions list must be monotonically increasing"

    assert region_type in REGION_TYPES, \
        'Supported region types: {}'.format(REGION_TYPES)

    traj = np.asarray(traj)
    assert traj.ndim == 2, "`traj` must be a two dimensional array"

    persons = np.asarray(persons)
    assert persons.ndim == 3,\
        "`persons` must be a three dimensional array, indexed by frame"

    assert persons.shape[0] == traj.shape[0],\
        "No of frames must match in `traj` and `persons`"

    intimate, personal, social = 0, 0, 0
    for frame in range(traj.shape[0]):
        i, p, s = _personal_disturbance(traj[frame, :], persons[frame, :],
                                        region_type, regions)
        intimate += i
        personal += p
        social += s

    return intimate, personal, social


def _personal_disturbance(robot, persons, region_type='uniform',
                          regions=[0.45, 1.2, 3.6]):
    """ Compute the instantaneous personal disturbance

    """
    ic, pc, sc = 0, 0, 0
    for person in persons:
        if region_type == 'uniform':
            distance = edist(robot, person)

            # - cascade check for inclusive lazy count
            if distance < regions[2]:
                sc += 1

                if distance < regions[1]:
                    pc += 1

                    if distance < regions[0]:
                        ic += 1

        elif region_type == 'anisotropic':
            ed = edist(person, robot)

            # - cascade check for inclusive lazy count
            if ed < adist(person, robot, ak=regions[2]):
                sc += 1

                if ed < adist(person, robot, ak=regions[1]):
                    pc += 1

                    if ed < adist(person, robot, ak=regions[0]):
                        ic += 1

    return ic, pc, sc


def relation_disturbance(traj, relations):
    """ Compute the total disturbance caused on an relation zone

    """
    phi = 0.0
    for waypoint in traj:
        for relation in relations:
            dist, inside = distance_to_segment(waypoint, relation)
            if inside and dist < 0.6:
                phi += 1.0

    return phi
