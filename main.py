from vision.triangulation import TriangulationMethod, triangulate_points
from file_handler import Observation, world_from_file
from vision import CameraModel
from bundleadj import bundle_adjustment

import argparse
import os
import re


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY,"data")
CAMERA_FILE = os.path.join(DATA_DIRECTORY,"camera.dat")
WORLD_FILE = os.path.join(DATA_DIRECTORY, "world.dat")
MEAS_REGEX = r"^meas-\d*\.dat$"

def main(args) -> None:
    iterations = args.iterations

    t_method = TriangulationMethod.ALL_OBSERVATIONS
    if args.method == "pair":
        t_method = TriangulationMethod.PAIR_OBSERVATIONS

    meas_files = [
        file for file in os.listdir(DATA_DIRECTORY)
        if re.match(MEAS_REGEX, file)
    ]

    observations = [
        Observation.from_file(os.path.join(DATA_DIRECTORY, filename))
        for filename in meas_files
    ]

    observations.sort(key=lambda x : x.sequence)

    p_world = world_from_file(WORLD_FILE)
    camera_model = CameraModel.from_file(CAMERA_FILE)

    print(f"{'Triangulating points':.^80}\n")

    p_triang = triangulate_points(
        camera_model=camera_model,
        observations=observations,
        method=t_method
    )
    true_points = {k: v for k,v in p_world.items() if k in p_triang}

    if not p_triang:
        print("Couldn't triangulate any point")
        return

    print(f"{len(p_triang)} points triangulated\n")

    print(f"{'Bundle adjustment using total least square':.^80}\n")

    bundle_adjustment(
        observations=observations,
        triangulated_points=p_triang,
        cameramodel=camera_model,
        true_points=true_points,
        iterations=iterations
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("-m", "--method", type=str,choices=["all", "pair"],
                        default="all")
    args = parser.parse_args()
    main(args)


