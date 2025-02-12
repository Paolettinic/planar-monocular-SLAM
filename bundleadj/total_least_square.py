from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from vision.cameramodel import CameraModel
from observation import Observation
from utils.utils import v2t_se2
from tqdm import tqdm
from .pose import linearize_poses
from .projection import linearize_projections
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def boxplus(
    x_r: NDArray,
    x_l: NDArray,
    size_dx_r: int,
    size_dx_l: int,
    delta_x: NDArray,
) -> Tuple[NDArray, NDArray]:
    """boxplus"""

    delta_x = delta_x.reshape(-1)
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
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:

    xr_size = size_dx_r * x_r.shape[0]
    xl_size = size_dx_l * x_l.shape[0]
    system_size = xr_size + xl_size

    chi_proj_stat = np.zeros(iterations)
    chi_pose_stat = np.zeros(iterations)

    inliers_p = np.zeros(iterations)

    t_iterations = tqdm(range(iterations), desc="TLS Iteration")

    for i in t_iterations:
        h = np.zeros((system_size, system_size))
        b = np.zeros((system_size, 1))

        h_proj, b_proj, chi_proj_stat[i], inliers_p[i] = linearize_projections(
            x_r=x_r,
            x_l=x_l,
            z=z_proj,
            size_dx_r=size_dx_r,
            size_dx_l=size_dx_l,
            proj_association=proj_association,
            camera_model=camera_model
        )

        h_pose, b_pose, chi_pose_stat[i] = linearize_poses(
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

        dx = np.zeros((system_size, 1))

        # keep the first pose fixed
        dx[size_dx_r :] = -np.linalg.solve(
            h[size_dx_r :, size_dx_r :], b[size_dx_r :]
        )

        error = np.linalg.norm(dx)
        t_iterations.set_postfix({"error":error})
        x_r, x_l = boxplus(x_r, x_l, size_dx_r, size_dx_l, dx)

    return x_r, x_l, chi_pose_stat, chi_proj_stat, inliers_p


def bundle_adjustment(
    observations: List[Observation],
    triangulated_points: Dict[int, NDArray],
    cameramodel: CameraModel,
    true_points: Dict[int, NDArray],
    iterations: int = 5
) -> None:

    point_to_index = {}
    index_to_point = {}

    num_poses = len(observations)
    num_points = len(triangulated_points)

    xr_guess = np.zeros((num_poses, 3, 3))

    xl_guess = np.zeros((num_points, 3))
    xl_gold = np.zeros((len(true_points), 3))

    for index, pt_id in enumerate(sorted(triangulated_points)):
        point_to_index[pt_id] = index
        index_to_point[index] = pt_id
        xl_guess[index, :] = triangulated_points[pt_id]
        xl_gold[index, :] = true_points[pt_id]

    z_odom = np.zeros((num_poses - 1, 3, 3))
    z_proj = np.zeros((num_poses * num_points, 2))

    pose_association = []
    proj_association = []

    true_positions = np.zeros((num_poses, 4))
    odom_positions = np.zeros((num_poses, 4))


    num_proj = 0

    for i in range(num_poses):
        xr_guess[i] = v2t_se2(np.array(
            [*observations[i].odom_pos, observations[i].odom_angle]
        ))

        if i > 0: # Create the odometry measure
            z_odom[i - 1] = np.linalg.inv(xr_guess[i - 1]) @ xr_guess[i]
            pose_association.append((i - 1, i))

        odom_positions[i] = np.array([
            *observations[i].odom_pos,
            np.cos(observations[i].odom_angle),
            np.sin(observations[i].odom_angle)
        ])
        true_positions[i] = np.array([
            *observations[i].gt_pos,
            np.cos(observations[i].gt_angle),
            np.sin(observations[i].gt_angle)
        ])

        for pt_id in observations[i].image_points:
            if pt_id not in triangulated_points:
                continue
            z_proj[num_proj, :] = np.array(
                [*observations[i].image_points[pt_id]]
            )
            proj_association.append((i, point_to_index[pt_id]))
            num_proj += 1

    z_proj = z_proj[:num_proj, :]


    x_r, x_l, chi_pose_stat, chi_proj_stat, proj_inliers = total_least_square(
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
    )

    figure = go.Figure()
    for i in range(x_l.shape[0]):
        figure.add_scatter3d(
            x=[xl_gold[i, 0], x_l[i, 0], xl_guess[i, 0]],
            y=[xl_gold[i, 1], x_l[i, 1], xl_guess[i, 1]],
            z=[xl_gold[i, 2], x_l[i, 2], xl_guess[i, 2]],
            mode="lines+markers",
            marker=dict(size=4,color=["green","orange","blue"])
        )
    figure.show()

    computed_poses = np.zeros((num_poses,4))
    for i in range(num_poses):
        computed_poses[i, :2] = x_r[i, :2, 2]
        computed_poses[i, 2:] = x_r[i, :2, 0]

    _, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2)


    ax1.title.set_text("odometry")
    ax1.quiver(
        odom_positions[:,0],
        odom_positions[:,1],
        odom_positions[:,2],
        odom_positions[:,3],
        color="blue",
    )
    ax1.quiver(
        true_positions[:,0],
        true_positions[:,1],
        true_positions[:,2],
        true_positions[:,3],
        color="green",
    )
    ax1.quiver(
        computed_poses[:,0],
        computed_poses[:,1],
        computed_poses[:,2],
        computed_poses[:,3],
        color="red",
    )
    ax1.legend(["Odometry","Ground Truth","Optimized"])

    ax2.title.set_text("projection inliers")
    ax2.plot(proj_inliers)

    ax3.title.set_text("chi poses")
    ax3.plot(chi_pose_stat)

    ax4.title.set_text("chi projections")
    ax4.set_yscale("log")
    ax4.plot(chi_proj_stat)

    plt.show()

