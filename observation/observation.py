from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Observation:
    sequence: int
    gt_position: Tuple[float, ...]
    gt_orientation: float
    odom_position: Tuple[float, ...]
    odom_orientation: float
    points: Dict[int, Tuple[float, float]]

    @classmethod
    def from_file(cls, filepath: str) -> "Observation":
        with open(filepath, "r") as file_p:
            sequence = int(
                file_p.readline()
                    .strip().split(":")[1]
                    .strip()
            )
            *gt_position, gt_orientation = (
                float(pose) for pose in
                file_p.readline()
                    .strip().split(":")[1]
                    .strip().split()
            )
            *odom_position, odom_orientation = (
                float(pose) for pose in
                file_p.readline()
                    .strip().split(":")[1]
                    .strip().split()
            )
            points = {}
            for line in file_p:
                if line.strip() == "":
                    continue
                values = line.strip().split()[2:]
                points[int(values[0])] = (float(values[1]), float(values[2]))

        return Observation(
            sequence,
            tuple(gt_position),
            gt_orientation,
            tuple(odom_position),
            odom_orientation,
            points
        )

