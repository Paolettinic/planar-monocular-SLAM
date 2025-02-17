import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from vision.cameramodel import CameraModel
from utils.geometry import v2t_se2
from tqdm import trange
from .pose import linearize_poses
from .projection import linearize_projections


def boxplus(
    x_r: NDArray,
    x_l: NDArray,
    size_dx_r: int,
    size_dx_l: int,
    delta_x: NDArray,
) -> Tuple[NDArray, NDArray]:
    """ boxplus operator implementation """

    num_poses = x_r.shape[0]
    num_landmarks = x_l.shape[0]

    delta_xr = delta_x[: num_poses * size_dx_r]
    delta_xl = delta_x[num_poses * size_dx_r :]

    dxr = np.array([
        v2t_se2(delta_xr[i * size_dx_r : i * size_dx_r + size_dx_r])
        for i in range(num_poses)
    ])

    dxl = delta_xl.reshape(num_landmarks, size_dx_l)

    return dxr @ x_r, dxl + x_l


def total_least_square(
    x_r: NDArray,
    x_l: NDArray,
    z_proj: NDArray,
    z_odom: NDArray,
    size_dx_r: int,
    size_dx_l:  int,
    proj_association: List[Tuple[int, int]],
    pose_association: List[Tuple[int, int]],
    camera_model: CameraModel,
    iterations: int = 20,
    damping: float = 1e-4
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Performs the total least squares optimization for a pose-projeciton
    problem

    Args:

        - x_r (`NDArray`): The array of robot poses in `SE(2)`, each
            represented as a `3x3` transformation matrix.
        - x_l (`NDArray`): The array of landmark positions in world coordinates
        - z_proj (`NDArray`): The landmark measurements in image coordinates
        - z_odom (`NDArray`): The odometry measurements between one pose and
            the next
        - size_dx_r (`int`): The size of the pose perturbation vector
        - size_dx_l (`int`): The size of the landmark perturbation vector
        - proj_association (`List[Tuple[int, int]]`): A list of associations
            between poses and landmarks; e.g. (n, m) means that the landmark m
            was observed in pose n
        - pose_association (`List[Tuple[int, int]]`): A list of associations
            between poses; (n, m) means that there's a measurement of the
            odometry from pose n to pose m.
        - camera_model (`CameraModel`): The camera model
        - iterations (`int`, optional): The maximum number of optimization
            iterations. Defaults to `20`.
        - damping (`float`, optional): The damping factor for stabilizing the
            system. Defaults to `1e-4`.

    Returns:
        `Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]`:
            - `NDArray`: The optimized robot poses.
            - `NDArray`: The optimized landmark positions.
            - `NDArray`: The history of pose residual over iterations.
            - `NDArray`: The history of projection residual over iterations.
            - `NDArray`: The history of pose inliners over iterations.
            - `NDArray`: The history of projection inliners over iterations.
    """

    xr_size = size_dx_r * x_r.shape[0]
    xl_size = size_dx_l * x_l.shape[0]
    system_size = xr_size + xl_size

    chi_proj_stat = np.zeros(iterations)
    chi_pose_stat = np.zeros(iterations)

    inliers_proj = np.zeros(iterations)
    inliers_pose = np.zeros(iterations)

    t_iterations = trange(iterations, desc="TLS Iteration")

    for i in t_iterations:
        h = np.zeros((system_size, system_size))
        b = np.zeros(system_size)
        dx = np.zeros(system_size)

        h_proj, b_proj, chi_proj_stat[i], inliers_proj[i] =\
            linearize_projections(
                x_r=x_r,
                x_l=x_l,
                z=z_proj,
                size_dx_r=size_dx_r,
                size_dx_l=size_dx_l,
                proj_association=proj_association,
                camera_model=camera_model
            )

        h_pose, b_pose, chi_pose_stat[i], inliers_pose[i] = linearize_poses(
            x_r=x_r,
            z=z_odom,
            size_dx_r=size_dx_r,
            pose_association=pose_association
        )

        h += h_proj
        b += b_proj

        h[:xr_size, :xr_size] += h_pose
        b[:xr_size] += b_pose

        h += np.eye(system_size) * damping

        # keep the first pose fixed
        dx[size_dx_r :] = -np.linalg.solve(
            h[size_dx_r :, size_dx_r :], b[size_dx_r :]
        )

        error = np.linalg.norm(dx)
        t_iterations.set_postfix({"error":error})
        x_r, x_l = boxplus(x_r, x_l, size_dx_r, size_dx_l, dx)

    return x_r, x_l, chi_pose_stat, chi_proj_stat, inliers_pose, inliers_proj



