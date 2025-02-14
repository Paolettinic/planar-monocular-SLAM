from typing import Tuple, List
from numpy.typing import NDArray
from utils.geometry import d_rot_z_0_se2
import numpy as np

def pose_error_and_jacobian(
    x_ri: NDArray,
    x_rj: NDArray,
    size_dx_r: int,
    z: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Computes the pose error and Jacobians for a relative SE(2) odometry
        measurement.

    Args:
        x_ri (`NDArray`): The first robot pose in SE(2), represented as a 3x3
            transformation matrix.
        x_rj (`NDArray`): The second robot pose in SE(2), represented as a 3x3
            transformation matrix.
        size_dx_r (`int`): The size of the perturbation vector for the robot
            pose.
        z (`NDArray`): The measured relative transformation between `x_ri` and
            `x_rj`, represented as a 3x3 transformation matrix.

    Returns:
        `Tuple[NDArray, NDArray, NDArray]`:
            - `NDArray`: The 6x1 error vector between the estimated and
                measured relative transformation.
            - `NDArray`: The 6x3 Jacobian of the error with respect to `x_ri`.
            - `NDArray`: The 6x3 Jacobian of the error with respect to `x_rj`.
    """
    ri = x_ri[:2, :2]
    ti = x_ri[:2, 2]
    rj = x_rj[:2, :2]
    tj = x_rj[:2, 2]

    z_hat = np.zeros((2, 3))
    z_hat[:2, :2] = ri.T @ rj
    z_hat[:2,  2] = ri.T @ (tj - ti)

    error = (z_hat - z[:2]).flatten("F")

    jacobian_rj = np.zeros((6, size_dx_r))
    jacobian_rj[4 :, : 2] = ri.T
    jacobian_rj[: 4, 2] = (ri.T @ d_rot_z_0_se2  @ rj).flatten("F")
    jacobian_rj[4 :, 2] = (ri.T @ d_rot_z_0_se2  @ tj).flatten("F")

    jacobian_ri = -jacobian_rj

    return error, jacobian_ri, jacobian_rj

def linearize_poses(
    x_r: NDArray,
    z: NDArray,
    size_dx_r: int,
    pose_association: List[Tuple[int, int]],
    kernel_threshold: float = 1e-1
) -> Tuple[NDArray, NDArray, float, int]:
    """
    Constructs the linearized system for the pose-pose constraint of the factor
    graph

    Args:
        - x_r (`NDArray`): The array of robot poses in `SE(2)`, each
            represented as a `3x3` transformation matrix.
        - z (`NDArray`): The odometry measurements between one pose and the
            next
        - size_dx_r (`int`): The size of the pose perturbation vector
        - pose_association (`List[Tuple[int, int]]`): A list of associations
            between poses; (n, m) means that there's a measurement of the
            odometry from pose n to pose m.
        - kernel_threshold (`float`, optional): The threshold for the robust
            kernel. Defaults to `0.1`.

    Returns:
        `Tuple[NDArray, NDArray, float, int]`:
            - `NDArray`: The matrix `H` of the linearized system.
            - `NDArray`: The vector `b` of the linearized system.
            - `float`: Residual
            - `int`: The number of inliers (measurements that fall within the
                kernel threshold).
    """
    xr_size = size_dx_r * x_r.shape[0]

    h = np.zeros((xr_size, xr_size))
    b = np.zeros(xr_size)

    chi = 0.0
    num_inliers= 0

    for i, z_odom in enumerate(z):
        omega_pose = np.eye(6)
        #omega_pose *= 1e3

        idx_i, idx_j = pose_association[i]

        cur_x_ri = x_r[idx_i]
        cur_x_rj = x_r[idx_j]

        e, j_xr_i, j_xr_j = pose_error_and_jacobian(
            x_ri=cur_x_ri,
            x_rj=cur_x_rj,
            z=z_odom,
            size_dx_r=size_dx_r
        )

        chi_ = e @ omega_pose @ e
        if chi_ > kernel_threshold:
            omega_pose *= np.sqrt(kernel_threshold / chi_)
            chi_ = kernel_threshold
        else:
            num_inliers += 1
        chi += chi_

        idx_pose_i = idx_i * size_dx_r
        idx_pose_j = idx_j * size_dx_r

        h[
            idx_pose_i : idx_pose_i + size_dx_r,
            idx_pose_i : idx_pose_i + size_dx_r
        ] += j_xr_i.T @ omega_pose @ j_xr_i
        h[
            idx_pose_i : idx_pose_i + size_dx_r,
            idx_pose_j : idx_pose_j + size_dx_r
        ] += j_xr_i.T @ omega_pose @ j_xr_j
        h[
            idx_pose_j : idx_pose_j + size_dx_r,
            idx_pose_i : idx_pose_i + size_dx_r
        ] += j_xr_j.T @ omega_pose @ j_xr_i
        h[
            idx_pose_j : idx_pose_j + size_dx_r,
            idx_pose_j : idx_pose_j + size_dx_r
        ] += j_xr_j.T @ omega_pose @ j_xr_j


        b[
            idx_pose_i : idx_pose_i + size_dx_r
        ] += j_xr_i.T @ omega_pose @ e
        b[
            idx_pose_j : idx_pose_j + size_dx_r
        ] += j_xr_j.T @ omega_pose @ e

    return h, b, float(chi), num_inliers
