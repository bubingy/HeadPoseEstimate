"""This module provides frontal face detection and face embedding extractor."""

# coding=utf-8
import json

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

from model.FaceAlignment3D.TDDFA import TDDFA
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
    try:
        boxes, probs, _ = model.detect(img)
    except Exception as e:
        print(e)
        return None

    if len(boxes) == 0:
        return None

    # since in our scenario, there is only one face in the image
    # we will choose the most likely one
    choosen_idx = np.argmax(probs)
    return boxes[choosen_idx]


def get_direction_from_landmarks(landmarks: list) -> np.ndarray:
    """Get the direction of face from landmarks.
    Args:
        landmarks: a set of facial key points.
    Returns:
        3 vector which can indicate face direction.
    """
    pca = PCA(n_components=2)
    pca.fit(landmarks[17:])
    direction_h = -pca.components_[1]
    if np.dot(direction_h, landmarks[45]-landmarks[36]) < 0:
        direction_h *= -1
    direction_h /= np.linalg.norm(direction_h)
    
    direction_v = pca.components_[0]
    if np.dot(direction_v, landmarks[30]-landmarks[8]) < 0:
        direction_v *= -1
    direction_v /= np.linalg.norm(direction_v)

    direction_d = np.cross(direction_h, direction_v)
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


def estimate_head_pose(image: Image.Image, 
                       face_roi: np.ndarray, 
                       model: TDDFA,
                       debug=False) -> np.ndarray:
    """Estimate head pose.
    Args:
        image: PIL Image object.
        model: Instance of FaceAlignment class.
        debug: if true, return more information.
    Returns:
        yaw, pitch, roll of the face
    """
    param_lst, roi_box_lst = model(image, [face_roi])
    ver_lst = model.recon_vers(param_lst, roi_box_lst)

    landmarks = ver_lst[0].T
    for i in range(len(landmarks)):
        landmarks[i][1] *= -1

    direction = get_direction_from_landmarks(landmarks)
    rotation_matrix = estimate_best_rotation(
        direction,
        np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
    )
    Euler_angle = get_euler_angles_from_rotation_matrix(rotation_matrix)
    Euler_angle *= np.array([-1, -1, 1])
    if debug:
        return Euler_angle, landmarks
    else:
        return Euler_angle


