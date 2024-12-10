import numpy as np

from numpy.typing import NDArray
from utils.rotations import Rz
from typing import Tuple

class CameraModel:
    def __init__(
        self,
        calibration_matrix: NDArray,
        cam_transform: NDArray,
        z_near: float,
        z_far: float,
        width: int,
        height: int
    ) -> None:
        self.calibration_matrix = calibration_matrix
        self.cam_transform = cam_transform
        #Inversion is performed just once, so it's acceptable
        self.inv_cam_transform = np.linalg.inv(cam_transform)
        self.z_near = z_near
        self.z_far = z_far
        self.width = width
        self.height = height


    def get_projection_matrix(
        self,
        position: Tuple[float, ...],
        rotation: float
    ) -> NDArray:
        """
        Get the projection matrix given the position and the orientation
        Args:
        - position: `Tuple[float, float]`, (x, y) position of the robot
        - rotation: `float`, rotation around z of the robot
        """
        assert len(position) == 2
        R = Rz(rotation)

        # Computing the inverse transformation, since [R|t] maps a point in the
        # robot frame into the world frame.
        Rt = np.eye(4)
        Rt[:3, :3] = R.T
        Rt[:3, 3] = -R.T @ np.array([*position,0]) # 0 imposes planar motion

        T = (self.inv_cam_transform @ Rt)[:3, :]
        return self.calibration_matrix @ T

    @classmethod
    def from_file(cls, filepath: str) -> "CameraModel":
        with open(filepath, "r") as file_p:
            next(file_p)
            calibration_matrix= np.array([
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

        return cls(
            calibration_matrix=calibration_matrix,
            cam_transform=camera_transform,
            z_near=z_near,
            z_far=z_far,
            width=width,
            height=height
        )


