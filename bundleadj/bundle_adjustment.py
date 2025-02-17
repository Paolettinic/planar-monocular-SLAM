import numpy as np
from file_handler import Observation
from typing import List, Dict, NamedTuple
from numpy.typing import NDArray
from vision import CameraModel
from .total_least_square import total_least_square


class BundleAdjustmentResult(NamedTuple):
    """
    Named Tuple containing:
        - `NDArray`: The optimized robot poses (`x_r`).
        - `NDArray`: The optimized landmark positions (`x_l`).
        - `NDArray`: The history of pose residual over iterations.
            (`chi_pose_stat`).
        - `NDArray`: The history of projection residual over iterations.
            (`chi_proj_stat`).
        - `NDArray`: The number of inliers for projection constraints
            (`proj_inliers`).
        - `NDArray`: The number of inliers for pose constraints
            (`pose_inliers`).
        - `Dict[int, int]`: mapping from landmark IDs to indices in `x_l`.
    """
    x_r: NDArray
    x_l: NDArray
    chi_pose_stat: NDArray
    chi_proj_stat: NDArray
    pose_inliers: NDArray
    proj_inliers: NDArray
    point_to_index: Dict[int, int]


def bundle_adjustment(
    observations: List[Observation],
    triangulated_points: Dict[int, NDArray],
    cameramodel: CameraModel,
    iterations: int = 20
) -> BundleAdjustmentResult:
    """
    Prepares the factor graph and solve the system using Total Least Squares

    Args:
        - observations (`List[Observation]`): A list of observations, each
            containing odometry poses and 2D image point measurements.
        - triangulated_points (`Dict[int, NDArray]`): A dictionary mapping
            landmark IDs to their estimated `3D` world coordinates.
        - cameramodel (`CameraModel`): The camera model containing intrinsic
            and extrinsic parameters.
        - iterations (`int`, optional): The number of optimization iterations.
            Defaults to `5`.

    Returns:
        `BundleAdjustmentResult`
    """
    point_to_index = {}

    num_poses = len(observations)
    num_points = len(triangulated_points)

    xr_guess = np.zeros((num_poses, 3, 3))
    xl_guess = np.zeros((num_points, 3))

    z_odom = np.zeros((num_poses - 1, 3, 3))
    z_proj = np.zeros((num_poses * num_points, 2))

    pose_association = []
    proj_association = []

    sorted_pt_ids = sorted(triangulated_points)

    for index, pt_id in enumerate(sorted_pt_ids):
        point_to_index[pt_id] = index
        xl_guess[index] = triangulated_points[pt_id]

    num_proj = 0

    for i in range(num_poses):
        xr_guess[i] = observations[i].odom_pose

        if i > 0: # Create the odometry measurement
            z_odom[i - 1] = np.linalg.inv(xr_guess[i - 1]) @ xr_guess[i]
            pose_association.append((i - 1, i))

        for pt_id in observations[i].image_points:
            if pt_id not in triangulated_points:
                continue
            # Create the projection measurement
            z_proj[num_proj] = np.array(
                [*observations[i].image_points[pt_id]]
            )
            proj_association.append((i, point_to_index[pt_id]))
            num_proj += 1

    z_proj = z_proj[:num_proj]


    return BundleAdjustmentResult(
        *total_least_square(
            x_r=xr_guess,
            x_l=xl_guess,
            z_proj=z_proj,
            z_odom=z_odom,
            size_dx_r=3,
            size_dx_l=3,
            proj_association=proj_association,
            pose_association=pose_association,
            camera_model=cameramodel,
            iterations=iterations
        ),
        point_to_index
    )

