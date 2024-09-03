from observation import Observation
import numpy as np
from dataclasses import dataclass
from utils import Rz

@dataclass
class CameraModel:
    camera_matrix: np.ndarray
    cam_transform: np.ndarray
    z_near: float
    z_far: float
    width: int
    height: int


    def get_projection_matrix(self, position: np.ndarray, rotation: float) -> np.ndarray:
        R = Rz(rotation)
        tvec = np.stack(position, np.
        Rt = np.hstack((R.T, (-R.T @ tvec ).reshape(-1, 1)))
        return camera_matrix @ Rt


    def triangulate(self,
                    observation1: Observation,
                    observation2: Observation) -> np.ndarray:

        #num_points_pos = points_pos1.shape[0]
        #pts_3d_hom = np.zeros((num_points_pos, 4))

        #for i in range(num_points_pos):
        #    A = np.zeros((4, 4))
        #    A[0] = points_pos1[i, 0] * proj_matrix1[2, :] - proj_matrix1[0, :]
        #    A[1] = points_pos1[i, 1] * proj_matrix1[2, :] - proj_matrix1[1, :]
        #    A[2] = points_pos2[i, 0] * proj_matrix2[2, :] - proj_matrix2[0, :]
        #    A[3] = points_pos2[i, 1] * proj_matrix2[2, :] - proj_matrix2[1, :]

        #    _, _, V = np.linalg.svd(A)
        #    X = V[-1]

        #    pts_3d_hom[i] = X

        #return pts_3d_hom
        return np.array([])

    @staticmethod
    def from_file(filepath: str) -> "CameraModel":
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
            z_near=z_near,
            z_far=z_far,
            width=width,
            height=height
        )

