from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from vision.cameramodel import CameraModel
from observation import Observation
from utils.utils import rot_x, rot_y, rot_z, rotation_matrix
from tqdm import tqdm
from .pose import linearize_poses
from .projection import linearize_projections
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def se2_to_se3_vec(vector:NDArray) -> NDArray:
    return np.array([vector[0],vector[1], 0, 0, 0, vector[2]])


def v2t(vector: NDArray) -> NDArray:
    """
    vector to transformation
    Args:
        - vector (`NDArray`) : 6d vector that parametrizes a SE(3)
            transformation
    """
    T = np.eye(4)
    T[:3,:3] = rotation_matrix(vector[3:])
    T[:3, 3] = vector[:3].reshape((1,3))
    return T

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

    d_xr = np.zeros((size_xr, 4, 4))
    for i in range(size_xr):
        dxr = v2t(delta_xr[i * size_dx_r : i * size_dx_r + size_dx_r])
        x_r[i, :, :] = dxr @ x_r[i, :, :]
    delta_xl = delta_xl.reshape(size_xl, size_dx_l)

    return x_r, x_l + delta_xl

#def boxplus(
#    x_r: NDArray,
#    x_l: NDArray,
#    size_dx_r: int,
#    size_dx_l: int,
#    delta_x: NDArray,
#) -> Tuple[NDArray, NDArray]:
#    """boxplus"""
#    delta_x = delta_x.reshape(-1)
#    size_xr = x_r.shape[0]
#    size_xl = x_l.shape[0]
#
#    delta_xr = delta_x[: size_xr * size_dx_r]
#    delta_xl = delta_x[size_xr * size_dx_r :]
#
#    d_xr = np.zeros((size_xr, 4, 4))
#    for i in range(size_xr):
#        d_xr[i, :, :] = v2t(se2_to_se3_vec(
#            delta_xr[i * size_dx_r : i * size_dx_r + size_dx_r]
#        ))
#    delta_xl = delta_xl.reshape(size_xl, size_dx_l)
#
#    return d_xr @ x_r, x_l + delta_xl


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
    damping: float = 1e-5
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:

    xr_size = dxr_size * x_r.shape[0]
    xl_size = dxl_size * x_l.shape[0]
    system_size = xr_size + xl_size

    chi_proj_stat = np.zeros(iterations)
    chi_pose_stat= np.zeros(iterations)

    inliers_p = np.zeros(iterations)

    for i in tqdm(range(iterations)):
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

    system_size = len(observations)
    num_points = len(triangulated_points)

    xr = np.zeros((system_size, 4, 4))
    xl = np.zeros((len(triangulated_points), 3))

    xl_gold = np.zeros((len(true_points), 3))
    for index, pt_id in enumerate(triangulated_points):
        point_to_index[pt_id] = index
        index_to_point[index] = pt_id
        xl[index] = triangulated_points[pt_id]

        xl_gold[index] = true_points[pt_id]

    z_odom = np.zeros((system_size - 1, 4, 4))
    z_proj = np.zeros((system_size * num_points, 2))

    pose_association = [] #np.zeros((system_size - 1, 2))
    proj_association = [] #np.zeros((system_size * num_points, 2))

    true_positions = np.zeros((system_size, 2))
    true_positions[0, :] = np.array(observations[0].gt_pos)

    odom_positions = np.zeros((system_size, 2))
    odom_positions[0, :] = np.array(observations[0].odom_pos)


    num_proj = 0

    for i in range(system_size):
        xr[i,:,:] = v2t(se2_to_se3_vec(np.array(
            [*observations[i].odom_pos, observations[i].odom_angle]
        )))
        if i > 0:
            z_odom[i - 1, :, :] = np.linalg.inv(xr[i - 1]) @ xr[i]
            pose_association.append((i - 1, i))
        odom_positions[i, :] = np.array(observations[i].odom_pos)
        true_positions[i, :] = np.array(observations[i].gt_pos)

        for pt_id in observations[i].image_points:
            if pt_id not in triangulated_points:
                continue
            z_proj[num_proj, :] = np.array(
                [*observations[i].image_points[pt_id]]
            )
            proj_association.append((i, point_to_index[pt_id]))
            num_proj += 1

    z_proj = z_proj[:num_proj, :]

    #dx = np.array(
    #    ([1, 1, 0] * system_size)  +
    #    ([2, 2, 2] * num_points)
    #).reshape((system_size+num_points)*3, 1)
    #x_r, x_l = boxplus(xr, xl, 3, 3, dx)

    x_r, x_l, chi_pose_stat, chi_proj_stat, proj_inliers = total_least_square(
        x_r=xr,
        x_l=xl,
        z_proj=z_proj,
        z_odom=z_odom,
        dxr_size=6,
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
        ),
        go.Scatter3d(
            x=x_l[:, 0],
            y=x_l[:, 1],
            z=x_l[:, 2],
            mode="markers"
        )
    ])
    figure.show()

    computed_poses = np.zeros((system_size,2))
    for i in range(system_size):
        computed_poses[i, :] = x_r[i, :2, 3]

    _, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2)


    ax1.title.set_text("odometry")
    ax1.plot(odom_positions[:,0], odom_positions[:,1])
    ax1.plot(computed_poses[:,0], computed_poses[:,1])
    ax1.plot(true_positions[:,0], true_positions[:,1])

    ax2.title.set_text("projection inliers")
    ax2.plot(proj_inliers)

    ax3.title.set_text("chi poses")
    ax3.plot(chi_pose_stat)

    ax4.title.set_text("chi projections")
    ax4.plot(chi_proj_stat)

    plt.show()

