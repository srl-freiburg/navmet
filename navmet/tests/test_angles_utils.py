
from numpy.testing import assert_equal

import numpy as np

from navmet.angle_utils import normalize
from navmet.angle_utils import addangles
from navmet.angle_utils import subangles


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
    assert_equal(subangles(2*np.pi, np.pi), np.pi)
    # pi - -pi/2 = pi/2
    assert_equal(subangles(np.pi, -np.pi/2.0), np.pi/2.0)
