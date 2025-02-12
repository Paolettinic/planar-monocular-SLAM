import numpy as np
from numpy.typing import NDArray

def v2t_se2(vector: NDArray) -> NDArray:
    """
    vector to transformation in se(2)
    Args:
        - vector (`NDArray`) : 3d vector that parametrizes a SE(2)
            transformation
    """
    T = np.eye(3)
    T[:2,:2] = rot_z_se2(vector[2])
    T[:2, 2] = vector[:2]
    return T

def v2t(vector: NDArray) -> NDArray:
    """
    vector to transformation
    Args:
        - vector (`NDArray`) : 6d vector that parametrizes a SE(3)
            transformation
    """
    T = np.eye(4)
    T[:3,:3] = rotation_matrix(vector[3:])
    T[:3, 3] = vector[:3]
    return T

def se2_to_se3_vec(vector:NDArray) -> NDArray:
    return np.array([vector[0],vector[1], 0, 0, 0, vector[2]])

def rotation_matrix(rvector: NDArray) -> NDArray:
    return rot_x(rvector[0]) @ rot_y(rvector[1]) @ rot_z(rvector[2])

def rot_x(angle: float) -> NDArray:
    return np.array([
        [1,     0,              0],
        [0,     np.cos(angle),  -np.sin(angle)],
        [0,     np.sin(angle),  np.cos(angle)]
    ])

def rot_y(angle: float) -> NDArray:
    return np.array([
        [np.cos(angle),     0,  np.sin(angle)],
        [0,                 1,  0],
        [-np.sin(angle),    0,  np.cos(angle)]
    ])

def rot_z(angle: float) -> NDArray:
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle),  0],
        [0,             0,              1]
    ])

def rot_z_se2(angle: float) -> NDArray:
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

d_rot_x_0 = np.array([
    [ 0, 0, 0],
    [ 0, 0,-1],
    [ 0, 1, 0]
])

d_rot_y_0 = np.array([
    [ 0, 0, 1],
    [ 0, 0, 0],
    [-1, 0, 0]
])

d_rot_z_0 = np.array([
    [ 0,-1, 0],
    [ 1, 0, 0],
    [ 0, 0, 0]
])
d_rot_z_0_se2 = np.array([
    [ 0,-1],
    [ 1, 0]
])

def skew(vector: NDArray) -> NDArray:
    assert vector.shape == (3,) or vector.shape == (3,1)
    return np.array([
        [ 0,         -vector[2], vector[1]],
        [ vector[2], 0,         -vector[0]],
        [-vector[1], vector[0],  0]
    ])
