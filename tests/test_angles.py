import math

import numpy as np

from angles import _angle_between_vectors


def test_angle_between_vectors_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    angle = _angle_between_vectors(v1, v2)
    assert math.isclose(angle, 90.0, rel_tol=1e-3)


def test_angle_between_vectors_same_direction():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])
    angle = _angle_between_vectors(v1, v2)
    assert math.isclose(angle, 0.0, rel_tol=1e-3)


def test_angle_between_vectors_opposite_direction():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    angle = _angle_between_vectors(v1, v2)
    assert math.isclose(angle, 180.0, rel_tol=1e-3)

