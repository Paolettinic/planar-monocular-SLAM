from utils.utils import skew
from vision.cameramodel import CameraModel
from typing import Tuple, List
from numpy.typing import NDArray
import numpy as np

def projection_error_and_jacobian(
    x_r: NDArray,
    x_l: NDArray,
    z:NDArray,
    camera: CameraModel
) -> Tuple[bool, NDArray, NDArray, NDArray]:
    jwl = np.zeros((3, 3))
    jwr = np.zeros((3, 6))
    error = np.zeros(2)
    x_r_c = camera.inv_cam_transform
    K = camera.intrinsic_matrix

    x_w_c = x_r_c @ np.linalg.inv(x_r) #cXr @ (wXr)^(-1)

    ir = x_w_c[:3, :3]
    it = -ir @ x_w_c[:3 , 3]

    p_cam = ir @ x_l + it # point in camera frame
    p_img = K @ p_cam # point in image
    fz = 1 / p_img[2]
    fz2 = fz ** 2
    z_hat = (p_img * fz)[:2]

    # visibility check
    if (
        p_img[2] < camera.z_near or
        p_img[2] > camera.z_far or
        z_hat[0] < 0 or
        z_hat[0] > camera.width or
        z_hat[1] < 0 or
        z_hat[1] > camera.height
    ):
        return False, error, jwr, jwl


    error = z_hat - z

    jacobian_proj = np.array([
        [fz, 0, -p_img[0] / fz2],
        [0, fz, -p_img[1] / fz2],
    ])

    jwl = ir

    jwr[:3, :3] = -ir
    jwr[:3, 3:] = ir @ skew(x_l)

    return True, error, jacobian_proj @ K @ jwr, jacobian_proj @ K @ jwl


#def projection_error_and_jacobian(
#    x_r: NDArray,
#    x_l: NDArray,
#    z:NDArray,
#    camera: CameraModel
#) -> Tuple[bool, NDArray, NDArray, NDArray]:
#    x_r_c = camera.inv_cam_transform
#    K = camera.intrinsic_matrix
#    error = np.zeros(2)
#
#    jxr = np.zeros((3, 3))
#    jxl = np.zeros((3, 3))
#
#    rt = x_r[:3, :3].T
#    it = -rt @ x_r[:3, 3]
#
#    pw = rt @ x_l + it
#
#    if pw[2] < 0:
#        return False, error, jxr, jxl
#
#    #proj_point, z_hat, visible = camera_model.project_pt_world(
#    #    point_world=x_l,
#    #    x_r_w=x_r
#    #)
#
#    #if not visible or proj_point[2] < 0:
#    #    return False, error, jxr, jxl
#
#    dRt_dz_0 = np.array([
#        [ 0, 1, 0],
#        [-1, 0, 0],
#        [ 0, 0, 0]
#    ])
#
#    pcam = x_r_c[:3,:3] @ pw + x_r_c[:3, 3]
#
#    pimg = K @ pcam
#    fz = 1 / pimg[2]
#    fz2 = fz ** 2
#    z_hat = (pimg * fz)[:2]
#
#    error = z_hat - z
#    jxl =  K @ x_r_c[:3, :3] @ rt
#
#    jxr = np.hstack((-np.eye(3,2), dRt_dz_0 @ x_l.reshape(3,1))) @ jxl
#
#    jacobian_proj = np.array([
#        [fz, 0, -pimg[0] / fz2],
#        [0, fz, -pimg[1] / fz2],
#    ])
#
#    return True, error, jacobian_proj @ jxr, jacobian_proj @ jxl


def linearize_projections(
    x_r: NDArray,
    x_l: NDArray,
    z: NDArray,
    size_dx_r: int,
    size_dx_l: int,
    proj_association: List[Tuple[int, int]],
    camera_model: CameraModel,
    kernel_threshold: float = 100
) -> Tuple[NDArray, NDArray, float, int]:
    xr_size = size_dx_r * x_r.shape[0]
    system_size = xr_size + size_dx_l * x_l.shape[0]

    h = np.zeros((system_size, system_size))
    b = np.zeros((system_size, 1))
    chi = 0.0
    num_inliers = 0
    omega_proj = np.eye(2) * 1e-6
    for i, proj in enumerate(z):

        idx_pose, idx_land = proj_association[i]
        cur_xr = x_r[idx_pose]
        cur_xl = x_l[idx_land]

        index_pose_matrix = idx_pose * size_dx_r
        index_land_matrix = xr_size + idx_land * size_dx_l

        valid, e, jxr, jxl = projection_error_and_jacobian(
            x_r=cur_xr,
            x_l=cur_xl,
            z=proj,
            camera=camera_model
        )

        if not valid:
            continue

        chi_ = e @ e

        if chi_ > kernel_threshold:
            e *= np.sqrt(kernel_threshold / chi_)
            chi_ = kernel_threshold
        else:
            num_inliers += 1

        chi += chi_

        h[
            index_pose_matrix : index_pose_matrix + size_dx_r,
            index_pose_matrix : index_pose_matrix + size_dx_r
        ] += jxr.T @ omega_proj @ jxr
        h[
            index_pose_matrix : index_pose_matrix + size_dx_r,
            index_land_matrix : index_land_matrix + size_dx_l
        ] += jxr.T @ omega_proj @ jxl
        h[
            index_land_matrix : index_land_matrix + size_dx_l,
            index_pose_matrix : index_pose_matrix + size_dx_r
        ] += jxl.T @ omega_proj @ jxr
        h[
            index_land_matrix : index_land_matrix + size_dx_l,
            index_land_matrix : index_land_matrix + size_dx_l
        ] += jxl.T @ omega_proj @ jxl

        b[
            index_pose_matrix : index_pose_matrix + size_dx_r
        ] += (jxr.T @ omega_proj @ e).reshape(6, 1)
        #] += (jxr.T @ omega_proj @ e).reshape(3, 1)
        b[
            index_land_matrix : index_land_matrix + size_dx_l
        ] += (jxl.T @ omega_proj @ e).reshape(3, 1)

    return h, b, float(chi), num_inliers

