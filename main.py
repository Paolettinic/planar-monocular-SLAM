from observation import Observation
import vision
from vision.triangulation import TriangulationMethod, triangulate_points
import plotly.graph_objects as go

import os
import re

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
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

def main() -> None:
    meas_files = sorted([
        file for file in os.listdir(DATA_DIRECTORY)
        if re.match(MEAS_REGEX, file)
    ])

    observations = [
        Observation.from_file(os.path.join(DATA_DIRECTORY, filename))
        for filename in meas_files
    ]

    p_world = world_from_file(WORLD_FILE)
    camera_model = vision.CameraModel.from_file(CAMERA_FILE)

    p_triang = triangulate_points(
        camera_model=camera_model,
        observations=observations,
        method=TriangulationMethod.ALL_OBSERVATIONS
    )

    if not p_triang:
        print("Couldn't triangulate any point")
        return

    figure = go.Figure()
    for p in p_triang:
        figure.add_scatter3d(
            x=[p_world[p][0], p_triang[p][0]],
            y=[p_world[p][1], p_triang[p][1]],
            z=[p_world[p][2], p_triang[p][2]],
            mode='markers+lines',
            marker_size=2,
            marker_color=['blue','green'],
            line_color='red',
            showlegend=False
        )
    figure.show()
    #plt_p_world = np.array(list(p_world[p] for p in p_triang))
    #plt_p_triang = np.array(list(p_triang[p] for p in p_triang))


    #figure = go.Figure(data=[
    #    go.Scatter3d(
    #        x=plt_p_world[:, 0],
    #        y=plt_p_world[:, 1],
    #        z=plt_p_world[:, 2],
    #        mode='markers',
    #        marker=dict(color=5,size=3)
    #    ),
    #    go.Scatter3d(
    #        x=plt_p_triang[:, 0],
    #        y=plt_p_triang[:, 1],
    #        z=plt_p_triang[:, 2],
    #        mode='markers',
    #        marker=dict(color=2,size=3)
    #    )]
    #)



if __name__ == "__main__":
    main()

