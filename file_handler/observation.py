from dataclasses import dataclass
from typing import Dict, Tuple
from utils.geometry import v2t_se2
from numpy.typing import NDArray
import numpy as np

@dataclass
class Observation:
    sequence: int
    #gt_pos: Tuple[float, ...]
    #gt_angle: float
    true_pose: NDArray
    odom_pose: NDArray
    #odom_pos: Tuple[float, ...]
    #odom_angle: float
    image_points: Dict[int, Tuple[float, float]]

    @classmethod
    def from_file(cls, filepath: str) -> "Observation":
        with open(filepath, "r") as file_p:
            sequence = int(file_p.readline() .strip().split(":")[1] .strip())
            values = file_p.readline().strip().split(":")[1].strip().split()
            *gt_xy_pos, gt_angle = (float(pose) for pose in values)
            gt_pose = v2t_se2(np.array([*gt_xy_pos, gt_angle]))
            values = file_p.readline().strip().split(":")[1].strip().split()
            *odom_xy_pos, odom_angle = (float(pose) for pose in values)
            odom_pose = v2t_se2(np.array([*odom_xy_pos, odom_angle]))
            points = {}
            for line in file_p:
                if line.strip() == "":
                    continue
                values = line.strip().split()[2:]
                points[int(values[0])] = (float(values[1]), float(values[2]))

        return cls(
            sequence=sequence,
            true_pose=gt_pose,
            odom_pose=odom_pose,
            image_points=points
        )

