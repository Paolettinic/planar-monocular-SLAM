from vision.triangulation import TriangulationMethod, triangulate_points
from bundleadj.total_least_square import *
import plotly.graph_objects as go

import argparse
import os
import re


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


    ## Reproject point
    #for observation in observations:
    #    for pt in observation.image_points:
    #        _,proj_pt,_ = camera_model.project_pt_world(
    #            p_world[pt],
    #            observation.gt_pos,
    #            observation.gt_angle
    #        )
    #        #if proj_pt != observation.image_points[pt]:
    #        observation.image_points[pt] = tuple(proj_pt)


    print("-"*30 + "Triangulating points" + "-"*30)
    p_triang = triangulate_points(
        camera_model=camera_model,
        observations=observations,
        method=t_method
    )

    if not p_triang:
        print("Couldn't triangulate any point")
        return

    print()
    print(len(p_triang), "points triangulated")
    print()
    #figure = go.Figure()

    #for p in p_triang:
    #    figure.add_scatter3d(
    #        x=[p_world[p][0],p_triang[p][0]],
    #        y=[p_world[p][1],p_triang[p][1]],
    #        z=[p_world[p][2],p_triang[p][2]],
    #        mode="lines+markers",
    #        marker=dict(
    #            size=4,
    #            color=["green","blue"]
    #        )
    #    )
    #figure.show()
    print("Bundle adjustment using total least square")


    true_points = {k: v for k,v in p_world.items() if k in p_triang}


    bundle_adjustment(observations, p_triang, camera_model, true_points, iterations)

if __name__ == "__main__":
    #from bundleadj.projection import projection_error_and_jacobian

    #camera = CameraModel.from_file(CAMERA_FILE)
    #x_r = np.eye(4)
    #x_l = np.array([2,0,0])
    #p_cam, p_img, visible = camera.project_pt_world(point_world=x_l, x_r_w=x_r)
    #print(f"{p_cam=}")
    #print(f"{p_img=}")
    #print(f"{visible=}")
    #print("_"*30)
    ##p_img += np.array([0.1,0.1])
    #projection_error_and_jacobian(x_r, x_l, p_img, camera)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", type=int, default=5)
    parser.add_argument("-m", "--method", type=str,choices=["all", "pair"], default="all")
    args = parser.parse_args()
    main(args)


