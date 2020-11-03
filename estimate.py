# coding=utf-8

import os
import json

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation

from utils.utils import get_mat_from_txt, save_data_into_js, \
    get_euler_angles_from_rotation_matrix
from model.face_alignment import FaceAlignment
from model.mtcnn import MTCNN
from model.deep_face import detect_face, estimate_head_pose, \
    get_direction_from_landmarks

if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    root_path = 'F:\\DATA\\Biwi_Kinect_Head_Pose_Database\\03'
    index = '00093'
    img_path = os.path.join(
        root_path, f'frame_{index}_rgb.png'
    )
    label_path = os.path.join(
        root_path, f'frame_{index}_pose.txt'
    )
    img = Image.open(img_path)
    mtx = get_mat_from_txt(label_path)
    label = get_euler_angles_from_rotation_matrix(mtx, 'zyx')
    # initialize three networks
    # mtcnn: face detection
    # face_alighment: get 68 face landmarks
    mtcnn = MTCNN(image_size=224, device=torch.device('cpu'))
    face_alighment = FaceAlignment('model/3DFAN4.pth.tar', 'model/depth.pth.tar')
    bound_box = None
    try:
        bound_box = detect_face(img, mtcnn)
    except Exception as e:
        print(e)
        exit(-1)

    # calculate Euler angle
    rotation, landmarks = estimate_head_pose(img, bound_box, face_alighment, True)
    
    print('label: ', label)
    print('predict: ', rotation)
    
    direction = get_direction_from_landmarks(landmarks)

    landmarks -= landmarks[30]
    landmarks = landmarks.tolist()

    arrows = []
    direction_h = direction[0]
    direction_v = direction[1]
    direction_d = direction[2]

    arrows = [
        {
            "position": landmarks[30],
            "direction": direction_h.tolist()
        },
        {
            "position": landmarks[30],
            "direction": direction_v.tolist()
        },
        {
            "position": landmarks[30],
            "direction": direction_d.tolist()
        }
    ]
    save_data_into_js(
        landmarks, 
        arrows, 
        'F:\\Workspace\\3DPlot\\js\\data.js'
    )
