from dataclasses import dataclass

@dataclass
class Observation:
    sequence: int
    gt_pose: tuple
    odom_position: tuple
    odom_orientation: float
    points: dict



    @staticmethod
    def from_file(filepath: str) -> "Observation":
        with open(filepath, "r") as file_p:
            sequence = int(
                file_p.readline()
                    .strip().split(":")[1]
                    .strip()
            )
            gt_pose = tuple(float(pose) for pose in
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
            gt_pose,
            tuple(odom_position),
            odom_orientation,
            points
        )

