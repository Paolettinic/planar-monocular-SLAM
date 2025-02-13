import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from file_handler import Observation
from typing import List, Dict
from numpy.typing import NDArray
from utils.utils import v2t_se2
from vision import CameraModel
from .total_least_square import total_least_square

def bundle_adjustment(
    observations: List[Observation],
    triangulated_points: Dict[int, NDArray],
    cameramodel: CameraModel,
    true_points: Dict[int, NDArray],
    iterations: int = 5
) -> None:

    point_to_index = {}

    num_poses = len(observations)
    num_points = len(triangulated_points)

    xr_guess = np.zeros((num_poses, 3, 3))

    xl_guess = np.zeros((num_points, 3))
    xl_gold = np.zeros((len(true_points), 3))

    for index, pt_id in enumerate(sorted(triangulated_points)):
        point_to_index[pt_id] = index
        xl_guess[index, :] = triangulated_points[pt_id]
        xl_gold[index, :] = true_points[pt_id]

    z_odom = np.zeros((num_poses - 1, 3, 3))
    z_proj = np.zeros((num_poses * num_points, 2))

    pose_association = []
    proj_association = []

    true_positions = np.zeros((num_poses, 4))
    odom_positions = np.zeros((num_poses, 4))


    num_proj = 0

    for i in range(num_poses):
        xr_guess[i] = v2t_se2(np.array(
            [*observations[i].odom_pos, observations[i].odom_angle]
        ))

        if i > 0: # Create the odometry measure
            z_odom[i - 1] = np.linalg.inv(xr_guess[i - 1]) @ xr_guess[i]
            pose_association.append((i - 1, i))

        odom_positions[i] = np.array([
            *observations[i].odom_pos,
            np.cos(observations[i].odom_angle),
            np.sin(observations[i].odom_angle)
        ])
        true_positions[i] = np.array([
            *observations[i].gt_pos,
            np.cos(observations[i].gt_angle),
            np.sin(observations[i].gt_angle)
        ])

        for pt_id in observations[i].image_points:
            if pt_id not in triangulated_points:
                continue
            z_proj[num_proj, :] = np.array(
                [*observations[i].image_points[pt_id]]
            )
            proj_association.append((i, point_to_index[pt_id]))
            num_proj += 1

    z_proj = z_proj[:num_proj, :]


    x_r, x_l, chi_pose_stat, chi_proj_stat, proj_inliers = total_least_square(
        x_r=xr_guess,
        x_l=xl_guess,
        z_proj=z_proj,
        z_odom=z_odom,
        size_dx_r=3,
        size_dx_l=3,
        proj_association=proj_association,
        pose_association=pose_association,
        camera_model=cameramodel,
        iterations=iterations
    )

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=xl_gold[:, 0],
                y=xl_gold[:, 1],
                z=xl_gold[:, 2],
                mode="markers",
                marker=dict(color="green", size=8, symbol="cross"),
                name="Ground truth"
            ),
            go.Scatter3d(
                x=x_l[:, 0],
                y=x_l[:, 1],
                z=x_l[:, 2],
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
    figure.show()
    figure.write_html("output/scatter.html", full_html=False, include_plotlyjs='cdn')

    computed_poses = np.zeros((num_poses,4))
    for i in range(num_poses):
        computed_poses[i, :2] = x_r[i, :2, 2]
        computed_poses[i, 2:] = x_r[i, :2, 0]

    fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2)
    fig.set_size_inches(10, 10)


    ax1.title.set_text("odometry")
    ax1.quiver(
        odom_positions[:,0],
        odom_positions[:,1],
        odom_positions[:,2],
        odom_positions[:,3],
        color="blue",
    )
    ax1.quiver(
        true_positions[:,0],
        true_positions[:,1],
        true_positions[:,2],
        true_positions[:,3],
        color="green",
    )
    ax1.quiver(
        computed_poses[:,0],
        computed_poses[:,1],
        computed_poses[:,2],
        computed_poses[:,3],
        color="red",
    )
    ax1.legend(["Odometry","Ground Truth","Optimized"])

    ax2.title.set_text("projection inliers")
    ax2.plot(proj_inliers)

    ax3.title.set_text("chi poses")
    ax3.plot(chi_pose_stat)

    ax4.title.set_text("chi projections")
    ax4.set_yscale("log")
    ax4.plot(chi_proj_stat)

    plt.savefig("output/out.svg", format="svg", dpi=80)
    plt.show()
