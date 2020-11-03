"""This module provides frontal face detection and face embedding extractor."""

# coding=utf-8
import json

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation

from model.face_alignment import FaceAlignment
from utils.utils import get_euler_angles_from_rotation_matrix

def compose_transformation(trans_sequence: list) -> transforms.transforms:
    """Define a pipeline-like object that contains one or more transformation.
    Args:
        trans_sequence: absolute path of the image.
    Returns:
        transforms: A pipeline-like transformation sequence.
    """
    return transforms.Compose(trans_sequence.append(transforms.ToTensor()))


def detect_face(img: Image.Image, model: torch.nn.Module) -> list:
    """Find the face in the image.
    Args:
        img: PIL Image object.
        model: A torch model for detect face.
    Returns:
        list: If a face is found.
        empty list: No face is found in the image.
    """
    # detect faces in the image, return -1 if there is no face detected
    boxes, probs = model.detect(img)
    if len(boxes) == 0:
        return []

    # since in our scenario, there is only one face in the image
    # we will choose the most likely one
    choosen_idx = np.argmax(probs)
    return boxes[choosen_idx]


def get_direction_from_landmarks(landmarks: list):
    """Get the direction of face from landmarks.
    Args:
        landmarks: a set of facial key points.
    Returns:
        3 vector which can indicate face direction.
    """
    # TODO: pca returns a rough result which isn't the optimal most of time.
    pca = PCA(n_components=2)
    pca.fit(landmarks[17:])
    direction_h = -pca.components_[1]
    if direction_h[0] < 0:
        direction_h *= -1
    direction_h /= np.linalg.norm(direction_h)
    
    direction_v = pca.components_[0]
    if direction_v[1] < 0:
        direction_v *= -1
    direction_v /= np.linalg.norm(direction_v)

    direction_d = np.cross(direction_h, direction_v)
    if direction_d[2] < 0:
        direction_d *= -1
    direction_d /= np.linalg.norm(direction_d)
    return np.array([direction_h, direction_v, direction_d])


def estimate_best_rotation(transformed, origin):
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


def estimate_head_pose(image: Image.Image, 
                       face_roi: np.ndarray, 
                       model: FaceAlignment,
                       debug=False) -> np.ndarray:
    """Estimate head pose.
    Args:
        image: PIL Image object.
        model: Instance of FaceAlignment class.
        debug: if true, return more information.
    Returns:
        yaw, pitch, roll of the face
    """
    image = np.array(image)
    landmarks = model.get_landmarks(image, face_roi)

    for i in range(len(landmarks)):
        landmarks[i][1] = -1 * landmarks[i][1]

    direction = get_direction_from_landmarks(landmarks)
    rotation_matrix = estimate_best_rotation(
        direction,
        np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
    )
    Euler_angle = get_euler_angles_from_rotation_matrix(rotation_matrix, 'zyx')
    Euler_angle *= np.array([-1, -1, 1])
    if debug:
        return Euler_angle, landmarks
    else:
        return Euler_angle


