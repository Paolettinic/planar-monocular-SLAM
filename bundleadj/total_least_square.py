from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from vision.cameramodel import CameraModel
from observation import Observation
from utils.utils import se2_to_se3_vec, v2t
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
    size_xr = x_r.shape[0]
    size_xl = x_l.shape[0]

    delta_xr = delta_x[: size_xr * size_dx_r]
    delta_xl = delta_x[size_xr * size_dx_r :]

    new_xr = np.zeros_like(x_r)
    new_xl = np.zeros_like(x_l)

    for i in range(size_xr):
        dxr_i = delta_xr[i * size_dx_r : i * size_dx_r + size_dx_r]
        if size_dx_r == 3:
            dxr_i = se2_to_se3_vec(
                delta_xr[i * size_dx_r : i * size_dx_r + size_dx_r]
            )
        dxr = v2t(dxr_i)
        new_xr[i, :, :] = dxr @ x_r[i, :, :]

    for i in range(size_xl):
        dxl_i = delta_xl[i*size_dx_l : i*size_dx_l + size_dx_l]
        new_xl[i, :] = x_l[i, :] + dxl_i
    #delta_xl = delta_xl.reshape(size_xl, size_dx_l)

    return new_xr, new_xl


def total_least_square(
    x_r: NDArray,
    x_l: NDArray,
    z_proj: NDArray,
    z_odom: NDArray,
    dxr_size: int,
    dxl_size:  int,
    proj_association: List[Tuple[int, int]],
    pose_association: List[Tuple[int, int]],
    camera_model: CameraModel,
    iterations: int = 5,
    damping: float = 1e-4
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:

    xr_size = dxr_size * x_r.shape[0]
    xl_size = dxl_size * x_l.shape[0]
    system_size = xr_size + xl_size

    chi_proj_stat = np.zeros(iterations)
    chi_pose_stat= np.zeros(iterations)

    inliers_p = np.zeros(iterations)
    t_iterations = tqdm(range(iterations), desc="TLS Iteration")

    for i in t_iterations:
        h = np.zeros((system_size, system_size))
        b = np.zeros((system_size, 1))

        h_proj, b_proj, chi_proj_stat[i], inliers_p[i] = linearize_projections(
            x_r=x_r,
            x_l=x_l,
            z=z_proj,
            size_dx_r=dxr_size,
            size_dx_l=dxl_size,
            proj_association=proj_association,
            camera_model=camera_model
        )

        h_pose, b_pose, chi_pose_stat[i] = linearize_poses(
            x_r=x_r,
            z=z_odom,
            size_dx_r=dxr_size,
            pose_association=pose_association
        )

        h += h_proj
        b += b_proj

        h[:xr_size, : xr_size] += h_pose
        b[:xr_size] += b_pose

        h += np.eye(system_size) * damping

        dx = np.zeros((system_size, 1))

        dx[dxr_size :] = -np.linalg.solve(
            h[dxr_size :, dxr_size :], b[dxr_size :]
        )
        error = np.linalg.norm(dx)
        t_iterations.set_postfix({"error":f"{error:5.3}"})
        x_r, x_l = boxplus(x_r, x_l, dxr_size, dxl_size, dx)

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

    xr = np.zeros((num_poses, 4, 4))
    xl = np.zeros((len(triangulated_points), 3))

    xl_gold = np.zeros((len(true_points), 3))
    for index, pt_id in enumerate(sorted(triangulated_points)):
        point_to_index[pt_id] = index
        index_to_point[index] = pt_id
        xl[index] = triangulated_points[pt_id]
        xl_gold[index] = true_points[pt_id]

    z_odom = np.zeros((num_poses - 1, 4, 4))
    z_proj = np.zeros((num_poses * num_points, 2))

    pose_association = []
    proj_association = []

    true_positions = np.zeros((num_poses, 4))
    true_positions[0, :] = np.array([
        *observations[0].gt_pos,
        np.cos(observations[0].gt_angle),
        np.sin(observations[0].gt_angle),
    ])

    odom_positions = np.zeros((num_poses, 4))
    odom_positions[0, :] = np.array([
        *observations[0].odom_pos,
        np.cos(observations[0].odom_angle),
        np.sin(observations[0].odom_angle),
    ])


    num_proj = 0

    for i in range(num_poses):
        xr[i,:,:] = v2t(se2_to_se3_vec(np.array(
            [*observations[i].odom_pos, observations[i].odom_angle]
            #[*observations[i].gt_pos, observations[i].gt_angle]
        )))
        if i > 0:
            z_odom[i - 1, :, :] = np.linalg.inv(xr[i - 1]) @ xr[i]
            pose_association.append((i - 1, i))
        odom_positions[i, :] = np.array([
            *observations[i].odom_pos,
            np.cos(observations[i].odom_angle),
            np.sin(observations[i].odom_angle)
        ])
        true_positions[i, :] = np.array([
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
        x_r=xr,
        x_l=xl,
        z_proj=z_proj,
        z_odom=z_odom,
        dxr_size=3,
        dxl_size=3,
        proj_association=proj_association,
        pose_association=pose_association,
        camera_model=cameramodel,
        iterations=iterations
    )

    figure = go.Figure(data=[
        go.Scatter3d(
            x=xl_gold[:, 0],
            y=xl_gold[:, 1],
            z=xl_gold[:, 2],
            mode="markers",
            marker=dict(size=4,color="blue"),
            name="GT"
        ),
        go.Scatter3d(
            x=x_l[:, 0],
            y=x_l[:, 1],
            z=x_l[:, 2],
            mode="markers",
            marker=dict(size=4,color="red", opacity=0.8),
            name="OPT"
        ),
        go.Scatter3d(
            x=xl[:, 0],
            y=xl[:, 1],
            z=xl[:, 2],
            mode="markers",
            marker=dict(size=4,color="green", opacity=0.8),
            name="TRIANG"
        ),
    ])
    figure = go.Figure()
    for i in range(x_l.shape[0]):
        figure.add_scatter3d(
            x=[xl_gold[i,0], x_l[i,0], xl[i,0]],
            y=[xl_gold[i,1], x_l[i,1], xl[i,1]],
            z=[xl_gold[i,2], x_l[i,2], xl[i,2]],
            mode="lines+markers",
            marker=dict(size=4,color=["blue","red","green"])
        )
    figure.show()

    computed_poses = np.zeros((num_poses,4))
    for i in range(num_poses):
        computed_poses[i, :2] = x_r[i, :2, 3]
        computed_poses[i, 2:] = x_r[i, :2, 0]

    #_, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2)
    _, ax1 = plt.subplots(1,1)


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
    #ax1.plot(odom_positions[:,0], odom_positions[:,1])
    #ax1.plot(computed_poses[:,0], computed_poses[:,1])
    #ax1.plot(true_positions[:,0], true_positions[:,1])

    #ax2.title.set_text("projection inliers")
    #ax2.plot(proj_inliers)

    #ax3.title.set_text("chi poses")
    #ax3.plot(chi_pose_stat)

    #ax4.title.set_text("chi projections")
    #ax4.plot(chi_proj_stat)

    plt.show()

