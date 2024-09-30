from observation import Observation
import camera
import matplotlib.pyplot as plt

import os
import re

import numpy as np


BASE_DIRECTORY = os.path.curdir
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY,"data")
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

def main():

    meas_files = sorted([
        file for file in os.listdir(DATA_DIRECTORY)
        if re.match(MEAS_REGEX, file)
    ])

    observations = [
        Observation.from_file(os.path.join(DATA_DIRECTORY, filename))
        for filename in meas_files
    ]

    world = world_from_file(WORLD_FILE)
    camera_model = camera.from_file(CAMERA_FILE)

    triangulated_points = camera.triangulate_two_observations(
        camera_model,
        observations[0],
        observations[1]
    )

    for p_id in sorted(triangulated_points.keys()):
        print("W:\t",p_id," -> ", world[p_id])
        print("T:\t",p_id," -> ",triangulated_points[p_id])


if __name__ == "__main__":
    main()

