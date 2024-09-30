from observation import Observation
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from utils.rotations import Rz
from typing import Dict, List

@dataclass
class CameraModel:
    camera_matrix: NDArray
    cam_transform: NDArray
    inv_cam_transform: NDArray
    z_near: float
    z_far: float
    width: int
    height: int


    def get_projection_matrix(
        self,
        position: tuple,
        rotation: float
    ) -> NDArray:
        """
        Get the projection matrix given the position and the orientation
        Args:
        - position: np.ndarray
        - rotation: float
        """
        R = Rz(rotation)

        Rt = np.eye(4)
        Rt[:3, :3] = R.T
        Rt[:3, 3] = -R.T @ np.array([*position,0])
        T = (self.inv_cam_transform @ Rt)[:3, :]
        return self.camera_matrix @ T


def from_file(filepath: str) -> CameraModel:
    with open(filepath, "r") as file_p:
        next(file_p)
        camera_matrix= np.array([
            [
                float(value)
                for value in file_p.readline().strip().split()
            ]
            for _ in range(3)
        ])
        next(file_p)
        camera_transform = np.array([
            [
                float(value)
                for value in file_p.readline().strip().split()
            ]
            for _ in range(4)
        ])
        z_near = float(file_p.readline().split(":")[1].strip())
        z_far = float(file_p.readline().split(":")[1].strip())
        width = int(file_p.readline().split(":")[1].strip())
        height = int(file_p.readline().split(":")[1].strip())

    return CameraModel(
        camera_matrix=camera_matrix,
        cam_transform=camera_transform,
        inv_cam_transform=np.linalg.inv(camera_transform), # inversion is done just once, so it's accettable
        z_near=z_near,
        z_far=z_far,
        width=width,
        height=height
    )
def triangulate_all_observations(camera_model: CameraModel, observations: List[Observation]) -> NDArray:
    return np.array([])

def triangulate_two_observations(
    camera_model: CameraModel,
    observation1: Observation,
    observation2: Observation
) -> Dict[int, NDArray]:
    """
    Triangulates the 3D coordinates of points observed from two different
    positions.

    Args:
        camera_model (CameraModel): The camera model used to obtain projection
            matrices for each observation.
        observation1 (Observation): The first observation
        observation2 (Observation): The second observation

    Returns:
        Dict[int, NDArray]: A dictionary where the keys are the point IDs
        common to both observations and the values are the triangulated 3D
        coordinates of those points.

    """

    proj_matrix1 = camera_model.get_projection_matrix(
        observation1.odom_position,
        observation1.odom_orientation
    )

    proj_matrix2 = camera_model.get_projection_matrix(
        observation2.odom_position,
        observation2.odom_orientation
    )
    common_points = observation1.points.keys() & observation2.points.keys()

    triang_points = {}

    for point_id in common_points:
        u1, v1 = observation1.points[point_id]
        u2, v2 = observation2.points[point_id]

        A = np.array([
            u1 * proj_matrix1[2, :] - proj_matrix1[0, :],
            v1 * proj_matrix1[2, :] - proj_matrix1[1, :],
            u2 * proj_matrix2[2, :] - proj_matrix2[0, :],
            v2 * proj_matrix2[2, :] - proj_matrix2[1, :]
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1]
        X /= X[3]
        triang_points[point_id] = X[:3]
    return triang_points



