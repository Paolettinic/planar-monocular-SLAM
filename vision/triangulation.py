from .cameramodel import CameraModel
from typing import List, Dict
from observation import Observation
from numpy.typing import NDArray
from enum import Enum, auto
import itertools
import numpy as np



class TriangulationMethod(Enum):
    PAIR_OBSERVATIONS = auto()
    ALL_OBSERVATIONS = auto()



def triangulate_point(
    point_id: int,
    camera_model: CameraModel,
    observations: List[Observation]
) -> NDArray:
    triang_point = np.array([-1,-1,-1])
    A = np.zeros((2 * len(observations), 4))
    for i,observation in enumerate(observations):
        proj_matrix = camera_model.get_projection_matrix(
            observation.odom_position,
            observation.odom_orientation
        )
        assert point_id in observation.points

        u,v = observation.points[point_id]

        A[2*i, :] = u * proj_matrix[2, :] - proj_matrix[0, :]
        A[2*i + 1, :] = v * proj_matrix[2, :] - proj_matrix[1, :]

    # Computer Vision: Algorithms and Applications, Szeliski
    # Chap. 7.1 eq. 7.5, 7.6
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X /= X[3]
    triang_point = X[:3]

    return triang_point


def triangulate_two_observations(
    camera_model: CameraModel,
    observation1: Observation,
    observation2: Observation
) -> Dict[int, NDArray]:
    """
    Triangulates the 3D coordinates of points observed from two different
    positions.

    Args:
        - camera_model: `CameraModel`, The camera model used to obtain
            projection matrices for each observation.
        - observation1: `Observation`, The first observation
        - observation2: `Observation`, The second observation

    Returns:
        - `Dict[int, NDArray]`: A dictionary where the keys are the point IDs
            common to both observations and the values are the triangulated 3D
            coordinates of those points.
    """

    triang_points = {}
    common_points = observation1.points.keys() & observation2.points.keys()
    if common_points:

        for point_id in common_points:
            triang_points[point_id] = triangulate_point(
                point_id=point_id,
                camera_model=camera_model,
                observations=[observation1, observation2]
            )

    return triang_points


def triangulate_points_from_all_observations(
    camera_model: CameraModel,
    observations: List[Observation]
) -> Dict[int, NDArray]:
    triang_points: Dict[int, NDArray] = dict()
    #point and indices of the observations that contain it
    points: Dict[int, List[int]] = dict()

    # Finding correspondences
    for i, observation in enumerate(observations):
        for p in observation.points:
            if p not in points:
                points[p] = []
            points[p].append(i)


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
    method: TriangulationMethod = TriangulationMethod.PAIR_OBSERVATIONS
) -> Dict[int, NDArray]:
    triang_points = dict()
    if method == TriangulationMethod.PAIR_OBSERVATIONS:
        result = dict()
        # TODO: very slow
        for obs1, obs2 in itertools.combinations(observations, 2):
            two_obs_points = triangulate_two_observations(
                camera_model=camera_model,
                observation1=obs1,
                observation2=obs2
            )
            for point in two_obs_points:
                if point not in triang_points:
                    result[point] = []
                result[point].append(
                    two_obs_points[point]
                )

        # Merging points averaging the resulting positions
        triang_points= {
            point: np.vstack(result[point]).mean(0)
            for point in result
        }

    else:
        triang_points = triangulate_points_from_all_observations(
            camera_model=camera_model,
            observations=observations
        )

    return triang_points

