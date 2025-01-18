from typing import Tuple, List
from numpy.typing import NDArray
from utils.utils import skew, d_rot_x_0, d_rot_y_0, d_rot_z_0
import numpy as np

def pose_error_and_jacobian(
    x_ri: NDArray,
    x_rj: NDArray,
    z: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    ri_t = x_ri[:3, :3].T
    rj = x_rj[:3, :3]

    tj = x_rj[:3, 3]

    z_hat = (np.linalg.inv(x_ri) @ x_rj)[:3, :]

    error = z_hat.flatten('F') - z[:3, :].flatten('F')

    dh_dx = np.zeros((12,1))
    dh_dx[9 :] = (ri_t @ np.array([1, 0, 0])).reshape(3, 1)

    dh_dy = np.zeros((12,1))
    dh_dy[9 :] = (ri_t @ np.array([0, 1, 0])).reshape(3, 1)

    dg_daz = np.zeros((3, 4))
    dg_daz[:3, :3] = ri_t @ d_rot_z_0 @ rj
    dg_daz[:3, 3] = ri_t @ d_rot_z_0 @ tj

    dh_daz = dg_daz.flatten('F').reshape(12, 1)

    jacobian_rj = np.hstack((dh_dx, dh_dy, dh_daz))
    jacobian_ri = -jacobian_rj

    return error, jacobian_ri, jacobian_rj

#def pose_error_and_jacobian(
#    x_ri: NDArray,
#    x_rj: NDArray,
#    z: NDArray
#) -> Tuple[NDArray, NDArray, NDArray]:
#    z_hat = (np.linalg.inv(x_ri) @ x_rj)[:3, :]
#    error = z_hat.flatten('F') - z[:3, :].flatten('F')
#
#    ri = x_ri[:3, :3]
#    rj = x_ri[:3, :3]
#    tj = x_rj[:3, 3]
#
#    d_r_x = (ri.T @ d_rot_x_0 @ rj).flatten('F').reshape(9,1)
#    d_r_y = (ri.T @ d_rot_y_0 @ rj).flatten('F').reshape(9,1)
#    d_r_z = (ri.T @ d_rot_z_0 @ rj).flatten('F').reshape(9,1)
#
#    j_xr_j = np.zeros((12, 6))
#
#    j_xr_j[:9,3:] = np.hstack((d_r_x, d_r_y, d_r_z))
#    j_xr_j[9:,:3] = ri.T
#    j_xr_j[9:,3:] = -ri.T @ skew(tj)
#
#    j_xr_i = -j_xr_j
#
#    return error, j_xr_i, j_xr_j



def linearize_poses(
    x_r: NDArray,
    z: NDArray,
    size_dx_r: int,
    pose_association: List[Tuple[int, int]],
    kernel_threshold: float = 1e-3
) -> Tuple[NDArray, NDArray, float]:
    xr_size = size_dx_r * x_r.shape[0]

    h = np.zeros((xr_size, xr_size))
    b = np.zeros((xr_size, 1))

    omega = np.eye(12)
    omega[:9, :9] *= 1e2

    chi = 0.0

    for i, meas in enumerate(z):
        idx_i, idx_j = pose_association[i]

        cur_x_ri = x_r[idx_i]
        cur_x_rj = x_r[idx_j]

        e, j_xr_i, j_xr_j = pose_error_and_jacobian(cur_x_ri, cur_x_rj, meas)

        chi_ = (e.T @  e).item()
        if chi_ > kernel_threshold:
            e *= np.sqrt(kernel_threshold / chi_)
            chi_ = kernel_threshold
        chi += chi_

        idx_pose_i = idx_i * size_dx_r
        idx_pose_j = idx_j * size_dx_r

        h[
            idx_pose_i : idx_pose_i + size_dx_r,
            idx_pose_i : idx_pose_i + size_dx_r
        ] += j_xr_i.T @ omega @ j_xr_i
        h[
            idx_pose_i : idx_pose_i + size_dx_r,
            idx_pose_j : idx_pose_j + size_dx_r
        ] += j_xr_i.T @ omega @ j_xr_j
        h[
            idx_pose_j : idx_pose_j + size_dx_r,
            idx_pose_i : idx_pose_i + size_dx_r
        ] += j_xr_j.T @ omega @ j_xr_i
        h[
            idx_pose_j : idx_pose_j + size_dx_r,
            idx_pose_j : idx_pose_j + size_dx_r
        ] += j_xr_j.T @ omega @ j_xr_j


        b[
            idx_pose_i : idx_pose_i + size_dx_r
        ] += (j_xr_i.T @ omega @ e).reshape(size_dx_r, 1)
        #] += (j_xr_i.T @ omega @ e).reshape(3, 1)
        b[
            idx_pose_j : idx_pose_j + size_dx_r
        ] += (j_xr_j.T @ omega @ e).reshape(size_dx_r, 1)
        #] += (j_xr_j.T @ omega @ e).reshape(3, 1)

    return h, b, chi
