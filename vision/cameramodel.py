import numpy as np

from numpy.typing import NDArray
from utils.utils import rot_z
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



    def get_extrinsic_matrix(
        self,
        position: Optional[Tuple[float, ...]] = None,
        rotation: Optional[float] = None,
        x_r_w: Optional[NDArray] = None
    ) -> NDArray:
        """
        Get the projection matrix given the position and the orientation
        Args:
        - position: `Tuple[float, float]`, (x, y) position of the robot
        - rotation: `float`, rotation around z of the robot
        """

        if position and rotation:
            assert len(position) == 2
            R = rot_z(rotation)
            # Computing the inverse transformation, since [R|t] maps a point from
            # robot frame into the world frame.
            Rt = np.eye(4)
            Rt[:3, :3] = R.T
            Rt[:3, 3] = -R.T @ np.array([*position,0]) # 0 imposes planar motion
            T = (self.inv_cam_transform @ Rt)[:3, :]
        elif x_r_w is not None:
            T = (self.inv_cam_transform @ np.linalg.inv(x_r_w))[:3, :]
        else:
            T = self.inv_cam_transform[:3, :]

        return T

    def get_projection_matrix(
        self,
        position: Optional[Tuple[float, ...]] = None,
        rotation: Optional[float] = None,
        x_r_w: Optional[NDArray] = None
    ) -> NDArray:
        """
        Get the projection matrix given the position and the orientation
        Args:
        - position: `Tuple[float, float]`, (x, y) position of the robot
        - rotation: `float`, rotation around z of the robot
        """
        T = self.get_extrinsic_matrix(position=position, rotation=rotation, x_r_w=x_r_w)
        return self.intrinsic_matrix @ T

    def project_pt_world(
        self,
        point_world: NDArray,
        position: Optional[Tuple[float, ...]] = None,
        rotation: Optional[float] = None,
        x_r_w: Optional[NDArray] = np.eye(4)
    )-> Tuple[NDArray, NDArray, bool]:
        """
        """
        p_img = np.array([-1,-1,-1])
        in_range = False
        in_frame = False

        extrinsics = self.get_extrinsic_matrix(
            position=position,
            rotation=rotation,
            x_r_w=x_r_w
        )

        p_cam = extrinsics @ np.append(point_world,1)
        p_img = self.intrinsic_matrix @ p_cam
        p_img /= p_img[2]


        in_range = self.z_near < p_cam[2] < self.z_far
        in_frame = (0 < p_img[0] < self.width) and (0 < p_img[1] < self.height)

        visible = in_range and in_frame

        return p_cam, p_img[:2], bool(visible)



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


