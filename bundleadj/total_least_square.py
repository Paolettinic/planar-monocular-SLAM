from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from vision.cameramodel import CameraModel
from observation import Observation
from utils import rotations
from tqdm import tqdm

def se2_to_se3_vec(vector:NDArray) -> NDArray:
    return np.array(
        [vector[0],vector[1], 0, 0, 0, vector[2]]
    )


def v2t(vector: NDArray) -> NDArray:
    """
    vector to transformation
    Args:
        - vector (`NDArray`) : 6d vector:
            - d_x
            - d_y
            - d_z
            - d_alpha_x
            - d_alpha_y
            - d_alpha_z
    """
    T = np.eye(4)
    T[:3,:3] = rotations.Rz(vector[5])
    T[:3, 3] = vector[:3].reshape((1,3))
    return T

def boxplus(Xr, Xl, delta_x, delta_l) -> Tuple[NDArray, NDArray]:
    s = delta_x.shape[0]
    Dx = np.zeros((s, 4, 4))
    for i in range(s):
        Dx[i, :, :] = v2t(delta_x[i])
    return Dx @ Xr, Xl + delta_l

def pose_error_and_jacobian(x_r1, x_r2, z) -> Tuple[bool, NDArray, NDArray]:
    valid = False

    return False, np.array([]), np.array([])

def projection_error_and_jacobian(
    x_r: NDArray,
    x_l: NDArray,
    z:NDArray,
    camera_model: CameraModel
) -> Tuple[bool, NDArray, NDArray, NDArray]:
    error = np.zeros(2)
    jacobian_xr = np.zeros((3, 3))
    jacobian_xl = np.zeros((3, 3))

    proj_point, z_hat, visible = camera_model.project_pt_world(
        point_world=x_l,
        x_r_w=x_r
    )

    if not visible:
        return False, error, jacobian_xr, jacobian_xl

    error = z_hat - z

    rt = x_r[:3, :3].T
    rc = camera_model.inv_cam_transform[:3, :3]
    dRt_dz_0 = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 0]
    ])

    jxr = np.zeros((3,3))
    jxr_dxy = -rc @ rt @ np.eye(3, 2)
    jxr_daz = rc @ rt @ dRt_dz_0 @ x_l

    jxl_dxl = rc @ rt

    jxr[:, :2] = jxr_dxy
    jxr[:, 2] = jxr_daz

    fz = 1 / proj_point[2]
    fz2 = fz ** 2
    jacobian_proj = np.array([
        [fz, 0, -proj_point[0] / fz2],
        [0, fz, -proj_point[0] / fz2],
    ])

    return True, error, jacobian_proj @ jxr, jacobian_proj @ jxl_dxl


def linearize_projections(
    x_r: NDArray,
    x_l: NDArray,
    z: NDArray,
    size_dx_r: int,
    size_dx_l: int,
    proj_association: List[Tuple[int, int]],
    camera_model: CameraModel
) -> Tuple[NDArray, NDArray]:
    xr_size = size_dx_r * x_r.shape[0]
    system_size = xr_size + size_dx_l * x_l.shape[0]

    h = np.zeros((system_size, system_size))
    b = np.zeros((system_size, 1))

    for i, proj in enumerate(z):

        pose_idx, land_idx = proj_association[i]
        cur_xr = x_r[pose_idx]
        cur_xl = x_l[land_idx]

        valid, e, jxr, jxl = projection_error_and_jacobian(
            x_r=cur_xr,
            x_l=cur_xl,
            z=proj,
            camera_model=camera_model
        )

        if not valid:
            continue


        pose_matrix_index = pose_idx * size_dx_r
        land_matrix_index = xr_size + land_idx * size_dx_l

        h[
            pose_matrix_index : pose_matrix_index + size_dx_r,
            pose_matrix_index : pose_matrix_index + size_dx_r
        ] += jxr.T @ jxr
        h[
            pose_matrix_index : pose_matrix_index + size_dx_r,
            land_matrix_index : land_matrix_index + size_dx_l
        ] += jxr.T @ jxl
        h[
            land_matrix_index : land_matrix_index + size_dx_l,
            pose_matrix_index : pose_matrix_index + size_dx_r
        ] += jxl.T @ jxr
        h[
            land_matrix_index : land_matrix_index + size_dx_r,
            land_matrix_index : land_matrix_index + size_dx_r
        ] += jxl.T @ jxl

        b[pose_matrix_index : pose_matrix_index + size_dx_r] += (jxr.T @ e).reshape(3, 1)
        b[land_matrix_index : land_matrix_index + size_dx_r] += (jxr.T @ e).reshape(3, 1)

    return h, b


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
    iterations: int = 1
):
    import matplotlib.pyplot as plt
    system_size = dxr_size * x_r.shape[0] + dxl_size * x_l.shape[0]
    h = np.zeros((system_size, system_size))
    b = np.zeros((system_size, 1))

    h_proj, b_proj = linearize_projections(
        x_r=x_r,
        x_l=x_l,
        z=z_proj,
        size_dx_r=dxr_size,
        size_dx_l=dxl_size,
        proj_association=proj_association,
        camera_model=camera_model
    )
    hmat = h_proj != 0
    plt.imshow(hmat)
    plt.show()

        #h_pose, b_pose = linearize_poses(xr, xl, z_odom, dxr_size, dxl_size, pose_association)



def bundle_adjustment(
    observations: List[Observation],
    triangulated_points: Dict[int, NDArray],
    cameramodel: CameraModel
) -> None:

    point_to_index = {}
    index_to_point = {}

    system_size = len(observations)
    num_points = len(triangulated_points)

    xr = np.zeros((system_size, 4, 4))

    xl = np.zeros((len(triangulated_points), 3))

    for index, pt_id in enumerate(triangulated_points):
        point_to_index[pt_id] = index
        index_to_point[index] = pt_id
        xl[index] = triangulated_points[pt_id]

    z_odom = np.zeros((system_size - 1, 4, 4))
    z_proj = np.zeros((system_size * num_points, 2))

    pose_association = [] #np.zeros((system_size - 1, 2))
    proj_association = [] #np.zeros((system_size * num_points, 2))

    xr[0,:,:] = v2t(se2_to_se3_vec(np.array(
        [*observations[0].odom_pos, observations[0].odom_angle]
    )))

    num_proj = 0

    for pt_id in observations[0].image_points:
        if pt_id in triangulated_points:
            z_proj[num_proj, :] = np.array([*observations[0].image_points[pt_id]])
            proj_association.append((0, point_to_index[pt_id]))
            num_proj += 1

    for i in range(1, system_size):
        xr[i,:,:] = v2t(se2_to_se3_vec(np.array(
            [*observations[i].odom_pos, observations[i].odom_angle]
        )))
        z_odom[i - 1, :, :] = np.linalg.inv(xr[i]) @ xr[i - 1]
        pose_association.append((i - 1, i))

        for pt_id in observations[i].image_points:
            if pt_id in triangulated_points:
                z_proj[num_proj, :] = np.array([*observations[i].image_points[pt_id]]) #TODO: CONVERT POSITION TO NUMPY ARRAY?
                proj_association.append((i, point_to_index[pt_id]))
                num_proj += 1

    z_proj = z_proj[:num_proj, :]

    total_least_square(
        x_r=xr,
        x_l=xl,
        z_proj=z_proj,
        z_odom=z_odom,
        dxr_size=3,
        dxl_size=3,
        proj_association=proj_association,
        pose_association=pose_association,
        camera_model=cameramodel,
        iterations=4
    )











