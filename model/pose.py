"""This module provides frontal face detection and face embedding extractor."""

# coding=utf-8

import numpy as np

from utils.utils import get_euler_angles_from_rotation_matrix


def naive_pca(data):
    """A simplified pca

    Args:
        data: input data
    Return:
        components
    """
    data -= np.mean(data, axis=0)
    _, _, vt = np.linalg.svd(data, full_matrices=False)
    return vt


def get_direction_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Get the direction of face from landmarks.
    Args:
        landmarks: a set of facial key points.
    Returns:
        3 vector which can indicate face direction.
    """
    components = naive_pca(landmarks[17:])
    direction_h = components[1]
    if np.dot(direction_h, landmarks[45]-landmarks[36]) < 0:
        direction_h *= -1
    direction_h /= np.linalg.norm(direction_h)

    direction_v = components[0]
    if np.dot(direction_v, landmarks[30]-landmarks[8]) < 0:
        direction_v *= -1
    direction_v /= np.linalg.norm(direction_v)

    direction_d = components[2]
    if np.dot(direction_d, landmarks[30] - (landmarks[31]+landmarks[35]) / 2) < 0:
        direction_d *= -1
    direction_d /= np.linalg.norm(direction_d)
    return np.array([direction_h, direction_v, direction_d])


def estimate_best_rotation(transformed: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Find optimal rotation between corresponding 3d points.

    Args:
        transformed: rotated points.
        origin: original points.
    Returns:
        Rotation matrix.
    """
    transformed = np.asarray(transformed)
    if transformed.ndim != 2 or transformed.shape[-1] != 3:
        raise ValueError("Expected input `transformed` to have shape (N, 3), "
                            "got {}".format(transformed.shape))
    origin = np.asarray(origin)
    if origin.ndim != 2 or origin.shape[-1] != 3:
        raise ValueError("Expected input `origin` to have shape (N, 3), "
                            "got {}.".format(origin.shape))

    if transformed.shape != origin.shape:
        raise ValueError("Expected inputs `transformed` and `origin` to have same shapes"
                            ", got {} and {} respectively.".format(
                            transformed.shape, origin.shape))

    H = np.einsum('ji,jk->ik', transformed, origin)
    u, s, vt = np.linalg.svd(H)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    return np.dot(u, vt)


def estimate_head_pose(landmarks: np.ndarray, debug=False) -> np.ndarray:
    """Estimate head pose.
    Args:
        landmarks: 3d face landmarks.
        debug: if true, return more information.
    Returns:
        yaw, pitch, roll of the face
    """
    direction = get_direction_from_landmarks(landmarks)
    rotation_matrix = estimate_best_rotation(direction, np.identity(3))
    euler_angle = get_euler_angles_from_rotation_matrix(rotation_matrix)
    euler_angle *= np.array([-1, -1, 1])
    if debug:
        return euler_angle, landmarks
    else:
        return euler_angle
