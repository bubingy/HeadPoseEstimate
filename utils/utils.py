# coding=utf-8

import re
import os
import json

import numpy as np

def get_mat_from_txt(file_path:str) -> np.ndarray:
    """ Get rotation matrix from txt file
    
    Read the label file and convert the rotation matrix 
    from string to numpy.ndarray
    
    param file_path: absolute path of the label file

    return: rotation matrix
    """
    content = []
    with open(file_path, 'r') as label:
        content = label.readlines()
    
    rotation = []
    for idx_row in range(3):
        row = map(
            float, 
            content[idx_row].replace('\n', '').strip().split(' ')
        )
        rotation.append(list(row))
    return np.array(rotation)


def get_euler_angles_from_rotation_matrix(matrix, seq):
    """Convert rotation matrix to Euler angles
    
    Args:
        matrix: rotation matrix

    return: Euler angles
    """
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    seq = seq.lower()
    _elementary_basis_vector = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    if extrinsic:
        seq = seq[::-1]

    if matrix.ndim == 2:
        matrix = matrix[None, :, :]
    num_rotations = matrix.shape[0]

    # Step 0
    # Algorithm assumes axes as column vectors, here we use 1D vectors
    n1 = _elementary_basis_vector[seq[0]]
    n2 = _elementary_basis_vector[seq[1]]
    n3 = _elementary_basis_vector[seq[2]]

    # Step 2
    sl = np.dot(np.cross(n1, n2), n3)
    cl = np.dot(n1, n3)

    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    offset = np.arctan2(sl, cl)
    c = np.vstack((n2, np.cross(n1, n2), n1))

    # Step 3
    rot = np.array([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ])
    res = np.einsum('...ij,...jk->...ik', c, matrix)
    matrix_transformed = np.einsum('...ij,...jk->...ik', res, c.T.dot(rot))

    # Step 4
    angles = np.empty((num_rotations, 3))
    # Ensure less than unit norm
    positive_unity = matrix_transformed[:, 2, 2] > 1
    negative_unity = matrix_transformed[:, 2, 2] < -1
    matrix_transformed[positive_unity, 2, 2] = 1
    matrix_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = np.arccos(matrix_transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    safe1 = (np.abs(angles[:, 1]) >= eps)
    safe2 = (np.abs(angles[:, 1] - np.pi) >= eps)

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = np.logical_and(safe1, safe2)
    angles[safe_mask, 0] = np.arctan2(matrix_transformed[safe_mask, 0, 2],
                                      -matrix_transformed[safe_mask, 1, 2])
    angles[safe_mask, 2] = np.arctan2(matrix_transformed[safe_mask, 2, 0],
                                      matrix_transformed[safe_mask, 2, 1])

    if extrinsic:
        # For extrinsic, set first angle to zero so that after reversal we
        # ensure that third angle is zero
        # 6a
        angles[~safe_mask, 0] = 0
        # 6b
        angles[~safe1, 2] = np.arctan2(matrix_transformed[~safe1, 1, 0]
                                       - matrix_transformed[~safe1, 0, 1],
                                       matrix_transformed[~safe1, 0, 0]
                                       + matrix_transformed[~safe1, 1, 1])
        # 6c
        angles[~safe2, 2] = -np.arctan2(matrix_transformed[~safe2, 1, 0]
                                        + matrix_transformed[~safe2, 0, 1],
                                        matrix_transformed[~safe2, 0, 0]
                                        - matrix_transformed[~safe2, 1, 1])
    else:
        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = np.arctan2(matrix_transformed[~safe1, 1, 0]
                                       - matrix_transformed[~safe1, 0, 1],
                                       matrix_transformed[~safe1, 0, 0]
                                       + matrix_transformed[~safe1, 1, 1])
        # 6c
        angles[~safe2, 0] = np.arctan2(matrix_transformed[~safe2, 1, 0]
                                       + matrix_transformed[~safe2, 0, 1],
                                       matrix_transformed[~safe2, 0, 0]
                                       - matrix_transformed[~safe2, 1, 1])

    # Step 7
    if seq[0] == seq[2]:
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = np.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
    else:
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = np.logical_or(angles[:, 1] < -np.pi / 2,
                                    angles[:, 1] > np.pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = np.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    # Step 8
    if not np.all(safe_mask):
        print("Gimbal lock detected. Setting third angle to zero since"
                      " it is not possible to uniquely determine all angles.")

    # Reverse role of extrinsic and intrinsic rotations, but let third angle be
    # zero for gimbal locked cases
    if extrinsic:
        angles = angles[:, ::-1]
    angles = np.rad2deg(angles[0])
    # angles *= np.array([-1, -1, 1])
    return angles


def save_data_into_js(points: list, arrows: list, js_path: str):
    plot_data = {}
    plot_data["points"] = points
    plot_data["arrows"] = arrows
    
    plot_data_str = json.dumps(plot_data)
    with open(js_path, 'w+') as f:
        f.write(f'var data={plot_data_str}')


def get_num_of_images(root_directory: str) -> int:
    """get the number of images
    
    param root_directory: directory where store images and label files.

    return: path of image and its label file.
    """
    img_ext = {'.png', '.jpg', '.bmp', '.jpeg'}
    num_of_images = 0
    for root, _, files in os.walk(root_directory):
        for f in files:
            if os.path.splitext(f)[-1] not in img_ext:
                continue
            img_path = os.path.join(root, f)
            if 'Biwi_Kinect_Head_Pose' in img_path:
                label_path = os.path.join(
                    root, f.replace('rgb.png', 'pose.txt')
                )
            elif '300W_LP' in img_path:
                label_path = os.path.join(
                    root, f.replace('.jpg', '.mat')
                )
            else:
                continue
            num_of_images += 1
    return num_of_images


def data_loader(root_directory: str):
    """Return a generator which give path of image and its label file
    
    param root_directory: directory where store images and label files.

    return: path of image and its label file.
    """
    img_ext = {'.png', '.jpg', '.bmp', '.jpeg'}
    for root, _, files in os.walk(root_directory):
        for f in files:
            if os.path.splitext(f)[-1] not in img_ext:
                continue
            img_path = os.path.join(root, f)
            if 'Biwi_Kinect_Head_Pose' in img_path:
                label_path = os.path.join(
                    root, f.replace('rgb.png', 'pose.txt')
                )
            elif '300W_LP' in img_path:
                label_path = os.path.join(
                    root, f.replace('.jpg', '.mat')
                )
            else:
                label_path = ''
            if os.path.exists(label_path) is False:
                continue
            yield (img_path, label_path)

