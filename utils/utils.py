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


def get_euler_angles_from_rotation_matrix(matrix):
    """Convert rotation matrix to Euler angles
    
    Args:
        matrix: rotation matrix
    return: 
        Euler angles
    """
    m00 = matrix[0][0]
    m02 = matrix[0][2]
    m10 = matrix[1][0]
    m11 = matrix[1][1]
    m12 = matrix[1][2]
    m20 = matrix[2][0]
    m22 = matrix[2][2]

    if m10 > 0.998:
        bank = 0
        attitude = np.pi/2
        heading = np.arctan2(m02, m22)
    elif m10 < -0.998:
        bank = 0
        attitude = -np.pi/2
        heading = np.arctan2(m02, m22)
    else:
        bank = np.arctan2(-m12, m11)
        attitude = np.arcsin(m10)
        heading = np.arctan2(-m20, m00)
    return  np.rad2deg(np.array([attitude, heading, bank]))
        

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

