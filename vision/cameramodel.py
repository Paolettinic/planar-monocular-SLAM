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
            T = (self.inv_cam_transform @ np.linalg.inv(x_r_w))[:3, :] #TODO: change to rt
        else:
            T = self.inv_cam_transform[:3, :]

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

        projection_matrix = np.eye(4)

        if position and rotation:
            projection_matrix = self.get_projection_matrix(
                position=position,
                rotation=rotation
            )
        else:
            projection_matrix = self.get_projection_matrix(x_r_w=x_r_w)

        projected_point = projection_matrix @ np.append(point_world,1)
        in_range = self.z_near < projected_point[2] < self.z_far
        if in_range:
            p_img = projected_point/projected_point[2]
            in_frame = (0 < p_img[0] < self.width) and (0 < p_img[1] < self.height)

        valid = in_range and in_frame

        return projected_point, p_img[:2], bool(valid)



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


