from .cameramodel import CameraModel
from typing import List, Dict
from observation import Observation
from numpy.typing import NDArray
from enum import Enum, auto
import itertools
import numpy as np



class TriangulationMethod(Enum):
    """Enum class for the two methods of triangulation
    Attributes:
        PAIR_OBSERVATIONS (auto): Triangulate points from pairwise observations
        ALL_OBSERVATIONS (auto): Triangulate points from all available observations
    """
    PAIR_OBSERVATIONS = auto()
    ALL_OBSERVATIONS = auto()



def triangulate_point(
    point_id: int,
    camera_model: CameraModel,
    observations: List[Observation]
) -> NDArray:
    """
    Triangulates the 3D coordinates of a point observed from different
    positions. The point must be present in every observation

    Args:
        point_id (`int`): The id of the point
        camera_model (`CameraModel`): The camera model used to obtain
            projection matrices for each observation.
        observations (`List[Observation]`): The list of the observation in
            which the point can be found

    Returns:
        `NDArray`: triangulated 3D coordinates of the point.

    Raises:
        `AssertionError`: the point is missing from one of the observations
    """
    A = np.zeros((2 * len(observations), 4))
    for i,observation in enumerate(observations):
        proj_matrix = camera_model.get_projection_matrix(
            observation.odom_pos,
            observation.odom_angle
        )

        # Test with ground truth position and orientation
        #proj_matrix = camera_model.get_projection_matrix(
        #    observation.gt_pos,
        #    observation.gt_angle
        #)

        assert point_id in observation.image_points,\
            f"One observation does not contain the point with id: {point_id}"

        u,v = observation.image_points[point_id]

        # Computer Vision: Algorithms and Applications, Szeliski
        # Chap. 7.1 eq. 7.5, 7.6
        A[2*i, :] = u * proj_matrix[2, :] - proj_matrix[0, :]
        A[2*i + 1, :] = v * proj_matrix[2, :] - proj_matrix[1, :]

    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X /= X[3]
    triang_point = X[:3]

    return triang_point


def triangulate_two_observations(
    camera_model: CameraModel,
    obs1: Observation,
    obs2: Observation
) -> Dict[int, NDArray]:
    """
    Triangulates the 3D coordinates of all points observable from two positions

    Args:
        camera_model (`CameraModel`): The camera model used to obtain
          projection matrices for each observation.
        obs1(`Observation`): The first observation
        obs2 (`Observation`): The second observation

    Returns:
        `Dict[int, NDArray]`: A dictionary where the keys are the point IDs
            common to both observations and the values are the triangulated 3D
            coordinates of those points. Empty dictionary is returned if there
            are not common points
    """

    triang_points = {}
    common_points = obs1.image_points.keys() & obs2.image_points.keys()
    if common_points:

        for point_id in common_points:
            triang_points[point_id] = triangulate_point(
                point_id=point_id,
                camera_model=camera_model,
                observations=[obs1, obs2]
            )

    return triang_points


def triangulate_points_from_all_observations(
    camera_model: CameraModel,
    observations: List[Observation]
) -> Dict[int, NDArray]:
    """
    Triangulates the 3D coordinates of points from all the available
    observations.

    Args:
        camera_model (`CameraModel`): The camera model used to obtain the
            projection matrices for each observation.
        observations (`List[Observation]`): List of all available observations

    Returns:
        `Dict[int, NDArray]`: A dictionary where the keys are the point IDs
            and the values are the triangulated 3D coordinates of those points.
            Empty dictionary is returned if no points are visible from at least
            two observations.
    """
    triang_points = dict()
    # point-> list of indices of the observations that contain said point
    points: Dict[int, List[int]] = dict()

    # Finding correspondences
    for i, observation in enumerate(observations):
        for p in observation.image_points:
            if p not in points:
                points[p] = []
            points[p].append(i)

    # Filter out points visible in only one observation
    points = {p: points[p] for p in points if len(points[p]) >= 2}

    # Triangulate point from all observation where a correspondence appears
    for point in points:
        triang_points[point] = triangulate_point(
            point_id=point,
            camera_model=camera_model,
            observations=[observations[i] for i in points[point]]
        )

    return triang_points


def triangulate_points(
    camera_model: CameraModel,
    observations: List[Observation],
    method: TriangulationMethod = TriangulationMethod.ALL_OBSERVATIONS
) -> Dict[int, NDArray]:
    triang_points = dict()
    try:
        # TODO: Check (very slow)
        if method == TriangulationMethod.PAIR_OBSERVATIONS:
            result = dict()

            for obs1, obs2 in itertools.combinations(observations, 2):
                two_obs_points = triangulate_two_observations(
                    camera_model=camera_model,
                    obs1=obs1,
                    obs2=obs2
                )

                for point in two_obs_points:
                    if point not in triang_points:
                        result[point] = []
                    result[point].append(two_obs_points[point])

            # Merging points averaging the resulting coordinates
            triang_points= {
                point: np.vstack(result[point]).mean(0)
                for point in result
            }

        else:
            triang_points = triangulate_points_from_all_observations(
                camera_model=camera_model,
                observations=observations
            )
    except AssertionError as msg:
        print(msg)
        return {}


    return triang_points

