import numpy as np

from numpy.typing import NDArray
from utils.geometry import rot_z, se2_to_se3
from typing import Tuple, Optional

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
        self.intrinsic_matrix = calibration_matrix
        self.cam_transform = cam_transform
        self.inv_cam_transform = np.linalg.inv(cam_transform)
        self.z_near = z_near
        self.z_far = z_far
        self.width = width
        self.height = height



    def get_extrinsic_matrix(self, x_r_w: NDArray) -> NDArray:
        """
        Get the projection matrix given the position and the orientation
        Args:
        - x_r_w (`NDArray`): The robot pose in SE(3), represented as a 4x4
            transformation matrix
        """
        return (self.inv_cam_transform @ np.linalg.inv(x_r_w))[:3, :]


    def get_projection_matrix(self, x_r_w: NDArray) -> NDArray:
        """
        Get the projection matrix given the position and the orientation
        Args:
        - position: `Tuple[float, float]`, (x, y) position of the robot
        - rotation: `float`, rotation around z of the robot
        """
        robot_position = x_r_w
        if x_r_w.shape == (3,3):
            robot_position = se2_to_se3(x_r_w)
        T = self.get_extrinsic_matrix(robot_position)
        return self.intrinsic_matrix @ T

    def project_point(
        self,
        point_world: NDArray,
        x_r_w: NDArray
    )-> Tuple[bool, NDArray, NDArray]:
        """
        """
        p_img = np.array([-1,-1,-1])
        in_range = False
        in_frame = False

        extrinsics = self.get_extrinsic_matrix(x_r_w)

        p_cam = extrinsics @ np.append(point_world,1)
        p_img = self.intrinsic_matrix @ p_cam
        p_img /= p_img[2]


        in_range = self.z_near < p_cam[2] < self.z_far
        in_frame = (0 < p_img[0] < self.width) and (0 < p_img[1] < self.height)

        visible = in_range and in_frame

        return bool(visible), p_cam, p_img[:2]



    @classmethod
    def from_file(cls, filepath: str) -> "CameraModel":
        with open(filepath, "r") as file_p:
            next(file_p)
            calibration_matrix = np.array([
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


