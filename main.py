from observation import Observation
from vision.cameramodel import CameraModel
from vision.triangulation import TriangulationMethod, triangulate_points
from bundleadj.total_least_square import *

import argparse
import os
import re


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY,"data")
#DATA_DIRECTORY = "/home/nico/Probabilistic/old_dataset/"
CAMERA_FILE = os.path.join(DATA_DIRECTORY,"camera.dat")
WORLD_FILE = os.path.join(DATA_DIRECTORY, "world.dat")
MEAS_REGEX = r"^meas-.*\.dat$"

def world_from_file(world_path) -> dict:
    world = {}
    with open(world_path, 'r') as world_file:
        for line in world_file:
            if line.strip() == "":
                continue
            p_id, *position = line.strip().split()
            world[int(p_id)] = [float(p) for p in position]
    return world

def main(args) -> None:
    iterations = args.iterations
    t_method= TriangulationMethod.ALL_OBSERVATIONS if args.method == "all" else TriangulationMethod.PAIR_OBSERVATIONS

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


    p_triang = triangulate_points(
        camera_model=camera_model,
        observations=observations,
        method=t_method
    )

    true_points = {k: v for k,v in p_world.items() if k in p_triang}

    if not p_triang:
        print("Couldn't triangulate any point")
        return

    bundle_adjustment(observations, p_triang, camera_model, true_points, iterations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", type=int, default=5)
    parser.add_argument("-m", "--method", type=str,choices=["all", "pair"], default="all")
    args = parser.parse_args()
    main(args)


