from utils.geometry import d_rot_z_0, se2_to_se3
from vision import CameraModel
from typing import Tuple, List
from numpy.typing import NDArray
import numpy as np


def projection_error_and_jacobian(
    x_r_se2: NDArray,
    x_l: NDArray,
    z:NDArray,
    size_dx_r: int,
    size_dx_l: int,
    camera: CameraModel
) -> Tuple[bool, NDArray, NDArray, NDArray]:
    """
    Computes the projection error and Jacobians for a landmark observed in a
    camera frame.

    Args:
        x_r_se2 (`NDArray`): The robot pose in SE(2) represented as a 3x3
            transformation matrix.
        x_l (`NDArray`): The landmark position in world coordinates (3D point).
        z (`NDArray`): The observed landmark in image coordinates.
        size_dx_r (`int`): The size of the perturbation vector for the robot
            pose.
        size_dx_l (`int`): The size of the perturbation vector for the landmark
            position.
        camera (`CameraModel`): The camera model

    Returns:
        `Tuple[bool, NDArray, NDArray, NDArray]`:
            - A boolean indicating whether the landmark is within the camera's
                field of view.
            - A 2D error vector representing the difference between projected
                and observed points.
            - The Jacobian of the error with respect to the robot pose.
            - The Jacobian of the error with respect to the landmark position.
    """
    x_r = se2_to_se3(x_r_se2)

    jwr = np.zeros((3, size_dx_r))
    jwl = np.zeros((3, size_dx_l))
    error = np.zeros(2)

    x_r_c = camera.inv_cam_transform
    K = camera.intrinsic_matrix

    x_w_c = x_r_c @ np.linalg.inv(x_r)
    ir = x_w_c[:3, :3]
    it = x_w_c[:3 , 3]

    p_cam = ir @ x_l + it # point in camera frame
    p_img = K @ p_cam # point in image frame
    fz = 1 / p_img[2]
    fz2 = fz ** 2
    z_proj = (p_img * fz)[:2]

    # visibility check
    if (
        p_cam[2] < camera.z_near or
        p_cam[2] > camera.z_far or

        z_proj[0] < 0 or
        z_proj[0] > camera.width or
        z_proj[1] < 0 or
        z_proj[1] > camera.height
    ):
        return False, error, jwr, jwl

    error = z_proj - z

    jacobian_proj = np.array([
        [fz, 0, -p_img[0] * fz2],
        [0, fz, -p_img[1] * fz2],
    ])
    d_rot_z_0_t = -d_rot_z_0  # d_rot_z_0 is skew -> negative = transpose

    jwr[:3, :2] = -ir @ np.eye(3,2)
    jwr[:3, 2] = ir @ d_rot_z_0_t @ x_l
    jwl = ir
    return True, error, jacobian_proj @ K @ jwr, jacobian_proj @ K @ jwl


def linearize_projections(
    x_r: NDArray,
    x_l: NDArray,
    z: NDArray,
    size_dx_r: int,
    size_dx_l: int,
    proj_association: List[Tuple[int, int]],
    camera_model: CameraModel,
    kernel_threshold: float = 1e3
) -> Tuple[NDArray, NDArray, float, int]:
    """
    Constructs the linearized system for the pose-projection constraint of the
    factor graph

    Args:
        - x_r (`NDArray`): The array of robot poses in `SE(2)`, each
            represented as a `3x3` transformation matrix.
        - x_l (`NDArray`): The array of landmark positions in world coordinates
        - z (`NDArray`): The landmark measurements in image coordinates
        - size_dx_r (`int`): The size of the pose perturbation vector
        - size_dx_l (`int`): The size of the landmark perturbation vector
        - proj_association (`List[Tuple[int, int]]`): A list of associations
            between poses and landmarks; e.g. (n, m) means that the landmark m
            was observed in pose n
        - camera_model (`CameraModel`): The camera model
        - kernel_threshold (`float`, optional): The threshold for the robust
            kernel. Defaults to `1000`.

    Returns:
        `Tuple[NDArray, NDArray, float, int]`:
            - `NDArray`: The matrix `H` of the linearized system.
            - `NDArray`: The vector `b` of the linearized system.
            - `float`: Residual
            - `int`: The number of inliers (measurements that fall within the
                kernel threshold).
    """
    xr_size = size_dx_r * x_r.shape[0]
    xl_size = size_dx_l * x_l.shape[0]
    system_size = xr_size + xl_size

    h = np.zeros((system_size, system_size))
    b = np.zeros(system_size)
    chi = 0
    num_inliers = 0

    for i, z_proj in enumerate(z):
        omega_proj = np.eye(2)

        idx_pose, idx_land = proj_association[i]
        cur_xr = x_r[idx_pose]
        cur_xl = x_l[idx_land]

        index_pose_matrix = idx_pose * size_dx_r
        index_land_matrix = xr_size + idx_land * size_dx_l

        valid, e, jxr, jxl = projection_error_and_jacobian(
            x_r_se2=cur_xr,
            x_l=cur_xl,
            z=z_proj,
            size_dx_r=size_dx_r,
            size_dx_l=size_dx_l,
            camera=camera_model
        )

        if not valid:
            continue

        chi_ = e @ omega_proj @ e

        if chi_ > kernel_threshold:
            omega_proj *= np.sqrt(kernel_threshold / chi_)
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
        ] += jxr.T @ omega_proj @ e
        b[
            index_land_matrix : index_land_matrix + size_dx_l
        ] += jxl.T @ omega_proj @ e


    return h, b, float(chi), num_inliers


