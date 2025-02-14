import numpy as np
from numpy.typing import NDArray

def compute_pose_error(xr_guess: NDArray, xr_true: NDArray):
    assert xr_guess.shape == xr_true.shape
    num_poses = xr_guess.shape[0]
    xr_size = xr_guess.shape[1]
    rel_t = np.zeros((num_poses - 1, xr_size, xr_size))
    rel_gt = np.zeros((num_poses - 1, xr_size, xr_size))
    for i in range(num_poses - 1):
        rel_t[i] = np.linalg.inv(xr_guess[i]) @ xr_guess[i+1]
        rel_gt[i] = np.linalg.inv(xr_true[i]) @ xr_true[i+1]
    error = np.linalg.inv(rel_t) @ rel_gt
    angle_error = np.atan2(error[:,1,0],error[:,0,0])
    position_error = error[:,:2,2]

    rmse_angle = np.sqrt(np.mean(angle_error ** 2))
    rmse_position = np.sqrt(np.mean(position_error ** 2, axis=0))
    return rmse_angle, rmse_position

def compute_landmark_error(xl_guess: NDArray, xl_true: NDArray):
    return np.sqrt(np.mean((xl_guess - xl_true)**2, axis=0))


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

def se2_to_se3(transformation: NDArray) -> NDArray:
    se3_tranformation = np.eye(4)
    se3_tranformation[:2,:2] = transformation[:2,:2]
    se3_tranformation[:2, 3] = transformation[:2, 2]
    return se3_tranformation

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
