from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Observation:
    sequence: int
    gt_pos: Tuple[float, ...]
    gt_angle: float
    odom_pos: Tuple[float, ...]
    odom_angle: float
    image_points: Dict[int, Tuple[float, float]]

    @classmethod
    def from_file(cls, filepath: str) -> "Observation":
        with open(filepath, "r") as file_p:
            sequence = int(file_p.readline() .strip().split(":")[1] .strip())
            values = file_p.readline().strip().split(":")[1].strip().split()
            *gt_pos, gt_angle = (float(pose) for pose in values)
            values = file_p.readline().strip().split(":")[1].strip().split()
            *odom_pos, odom_angle = (float(pose) for pose in values)
            points = {}
            for line in file_p:
                if line.strip() == "":
                    continue
                values = line.strip().split()[2:]
                points[int(values[0])] = (float(values[1]), float(values[2]))

        return cls(
            sequence=sequence,
            gt_pos=tuple(gt_pos),
            gt_angle=gt_angle,
            odom_pos=tuple(odom_pos),
            odom_angle=odom_angle,
            image_points=points
        )

