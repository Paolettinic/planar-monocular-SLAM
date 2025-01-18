import numpy as np
from numpy.typing import NDArray

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

def skew(vector: NDArray) -> NDArray:
    assert vector.shape == (3,) or vector.shape == (3,1)
    return np.array([
        [ 0,         -vector[2], vector[1]],
        [ vector[2], 0,         -vector[0]],
        [-vector[1], vector[0],  0]
    ])
