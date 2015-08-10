
from numpy.testing import assert_equal

import numpy as np

from navmet.utils import normalize
from navmet.utils import addangles
from navmet.utils import subangles
from navmet.utils import distance_to_segment


def test_norm_angle():
    assert_equal(normalize(3*np.pi), np.pi)
    assert_equal(normalize(np.pi/2.0), np.pi/2.0)
    assert_equal(normalize(-np.pi), np.pi)
    assert_equal(normalize(5*np.pi), np.pi)


def test_and_angles():
    # pi + pi = 0 or 2pi
    assert_equal(addangles(np.pi, np.pi), 0.0)
    # 2pi + pi = pi
    assert_equal(addangles(2*np.pi, np.pi), np.pi)
    # pi + -pi/2 = pi/2
    assert_equal(addangles(np.pi, -np.pi/2.0), np.pi/2.0)


def test_sub_angles():
    # pi - 2pi = pi
    assert_equal(subangles(np.pi, 2*np.pi), np.pi)
    # 2pi - pi = pi
    # assert_equal(subangles(2*np.pi, np.pi), np.pi)
    # pi - -pi/2 = pi/2
    # assert_equal(subangles(np.pi, -np.pi/2.0), np.pi/2.0)


def test_distance_to_segment():
    # test points
    x1 = (2.0, 2.0, 0, 0)  # colinear inside
    x2 = (4.0, 0.0, 0, 0)  # colinear outside
    x3 = (4.0, 1.0, 0, 0)  # outside not colinear
    x4 = (3.0, 3.0, 0, 0)  # inside not colinear
    x5 = (1.0, 2.0, 0, 0)  # inside not colinear
    x6 = (2.7, 2.7, 0, 0)  # inside not colinear

    # line
    ls = (1.0, 3.0)
    le = (3.0, 1.0)
    assert_equal(distance_to_segment(x1, (ls, le))[1], True)
    assert_equal(distance_to_segment(x1, (ls, le))[0], 0.0)

    assert_equal(distance_to_segment(x2, (ls, le))[1], False)

    assert_equal(distance_to_segment(x3, (ls, le))[1], False)

    assert_equal(distance_to_segment(x6, (ls, le))[1], True)
    assert_equal(distance_to_segment(x4, (ls, le))[1], True)
    assert_equal(distance_to_segment(x5, (ls, le))[1], True)
