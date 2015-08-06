
from numpy.testing import assert_approx_equal
from numpy.testing import assert_equal

import numpy as np

from navmet.angle_utils import normalize
from navmet.angle_utils import addangles
from navmet.angle_utils import subangles


def test_norm_angle():
    a1 = np.pi
    a2 = 3*np.pi
    assert_equal(normalize(a2), a1)
