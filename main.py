from vision.triangulation import TriangulationMethod, triangulate_points
from file_handler import Observation, world_from_file
from vision import CameraModel
from bundleadj import bundle_adjustment
from utils.geometry import compute_landmark_error, compute_pose_error
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import argparse
import os
import re

pio.kaleido.scope.default_width=900
pio.kaleido.scope.default_height=900

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

    odom_poses = np.array([
        observation.odom_pose
        for observation in observations
    ])

    true_poses = np.array([
        observation.true_pose
        for observation in observations
    ])

    p_world = world_from_file(WORLD_FILE)
    camera_model = CameraModel.from_file(CAMERA_FILE)

    """
    Initial solution: triangulating points with noisy odometry and refining
    the pose using TLS
    """

    print(f"{'Initial point triangulation':.^80}\n")


    initial_p_triang = triangulate_points(
        camera_model=camera_model,
        observations=observations,
        method=t_method
    )
    true_points = {k: v for k,v in p_world.items() if k in initial_p_triang}

    if not initial_p_triang:
        print("Couldn't triangulate any point")
        return

    print(f"{len(initial_p_triang)} points triangulated\n")

    print(f"{'Bundle adjustment using total least square':.^80}\n")

    result = bundle_adjustment(
        observations=observations,
        triangulated_points=initial_p_triang,
        cameramodel=camera_model,
        iterations=iterations
    )
    chi_pose_stat = result.chi_pose_stat
    chi_proj_stat = result.chi_proj_stat
    pose_inliers = result.pose_inliers
    proj_inliers = result.proj_inliers

    for i, observation in enumerate(observations):
        observation.odom_pose = result.x_r[i]

    """
    IMPROVED SOLUTION: Use the resulting pose state to obtain a better initial
    guess of the landmark position, leading to a better result
    """

    print(f"{'Triangulating points with updated poses':.^80}\n")

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

    result = bundle_adjustment(
        observations=observations,
        triangulated_points=p_triang,
        cameramodel=camera_model,
        iterations=iterations
    )
    chi_pose_stat = np.append(chi_pose_stat, result.chi_pose_stat)
    chi_proj_stat = np.append(chi_proj_stat, result.chi_proj_stat)
    pose_inliers = np.append(pose_inliers, result.pose_inliers)
    proj_inliers = np.append(proj_inliers, result.proj_inliers)

    for i, observation in enumerate(observations):
        observation.odom_pose = result.x_r[i]

    xl_true = np.zeros_like(result.x_l)
    xl_guess= np.zeros_like(result.x_l)
    for point in true_points:
        xl_true[result.point_to_index[point]] = true_points[point]
        xl_guess[result.point_to_index[point]] = initial_p_triang[point]

    # Computing resulting error
    rmse_angle, rmse_position = compute_pose_error(result.x_r, true_poses)
    landmark_error = compute_landmark_error(result.x_l, xl_true)
    print("Pose error")
    print(f"{'Rotation:':.<20}",f"{rmse_angle}")
    print(
        f"{'Translation:':.<20}",
        f"x:{rmse_position[0]:.>20}",
        f"y:{rmse_position[1]}"
    )
    print("Landmark_error")
    print("RMSE:",landmark_error[0], landmark_error[1], landmark_error[2])


    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=xl_true[:, 0],
                y=xl_true[:, 1],
                z=xl_true[:, 2],
                mode="markers",
                marker=dict(color="green", size=8, symbol="cross"),
                name="Ground truth"
            ),
            go.Scatter3d(
                x=result.x_l[:, 0],
                y=result.x_l[:, 1],
                z=result.x_l[:, 2],
                mode="markers",
                marker=dict(color="orange", size=8, symbol="circle-open"),
                name="Computed"
            ),
            go.Scatter3d(
                x=xl_guess[:, 0],
                y=xl_guess[:, 1],
                z=xl_guess[:, 2],
                mode="markers",
                marker=dict(color="blue", size=8, symbol="cross"),
                name="Triangulated points",
            ),
        ],
    )

    fig_odom, (ax1, ax2)= plt.subplots(1,2)
    fig_plots, ((ax3, ax4), (ax5, ax6))= plt.subplots(2,2)
    fig_odom.set_size_inches(10, 6)
    fig_plots.set_size_inches(10, 10)

    ax1.title.set_text("Odometry and GT")
    ax1.quiver(
        odom_poses[:, 0, 2],
        odom_poses[:, 1, 2],
        odom_poses[:, 0, 0],
        odom_poses[:, 1, 0],
        color="blue",
    )
    ax1.quiver(
        true_poses[:, 0, 2],
        true_poses[:, 1, 2],
        true_poses[:, 0, 0],
        true_poses[:, 1, 0],
        color="green",
    )
    ax1.legend(["Odometry","Ground Truth"])

    ax2.title.set_text("After optimization")
    ax2.quiver(
        odom_poses[:, 0, 2],
        odom_poses[:, 1, 2],
        odom_poses[:, 0, 0],
        odom_poses[:, 1, 0],
        color="blue",
    )
    ax2.quiver(
        true_poses[:, 0, 2],
        true_poses[:, 1, 2],
        true_poses[:, 0, 0],
        true_poses[:, 1, 0],
        color="green",
    )
    ax2.quiver(
        result.x_r[:, 0, 2],
        result.x_r[:, 1, 2],
        result.x_r[:, 0, 0],
        result.x_r[:, 1, 0],
        color="red",
    )
    ax2.legend(["Odometry","Ground Truth","Optimized"])

    ax3.title.set_text("pose inliers")
    ax3.plot(pose_inliers)

    ax4.title.set_text("projection inliers")
    ax4.plot(proj_inliers)

    ax5.title.set_text("chi poses")
    ax5.plot(chi_pose_stat)

    ax6.title.set_text("chi projections")
    ax6.set_yscale("log")
    ax6.plot(chi_proj_stat)

    fig_odom.savefig("output/odom.svg", format="svg", dpi=80)
    fig_plots.savefig("output/plot.svg", format="svg", dpi=80)
    plt.ion()
    plt.show(block=False)
    print("Optimization completed.")
    ans = input(
        "Do you want to visualize the landmarks in your browser? [Y/n]: "
    )
    if ans in ("Y","y",""):
        figure.show()
    print("SVG and HTML saved in the output folder")
    figure.write_image("output/scatter.svg")
    figure.write_html(
        "output/scatter.html",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", type=int, default=10)
    parser.add_argument("-m", "--method", type=str,choices=["all", "pair"],
                        default="all")
    args = parser.parse_args()
    main(args)


