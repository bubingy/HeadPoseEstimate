"""Test against AFLW2000"""
# coding=utf-8

import os
import time
import argparse
import webbrowser

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from model.FaceDetection.FaceBoxes import FaceBoxes
from model.FaceAlignment3D.TDDFA import TDDFA
from model.pose import estimate_head_pose, \
    get_direction_from_landmarks
from utils.utils import save_data_into_js, get_label_from_mat


if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    parser = argparse.ArgumentParser(description='estimate head pose.')
    parser.add_argument(
        '-i', '--image-path', required=True,
        help="path of image."
    )
    parser.add_argument(
        '-l', '--label-path', required=True,
        help="path of label file."
    )
    parser.add_argument(
        '-d', '--draw',
        action='store_true',
        default=False,
        help="whether to openbrowser to show landmarks and face orientation."
    )
    args = parser.parse_args()

    img_path = args.image_path
    img = Image.open(img_path)

    label_path = args.label_path
    label = get_label_from_mat(label_path)

    # initialize three networks
    # mtcnn: face detection
    # tddfa: get 68 3d face landmarks
    face_boxes = FaceBoxes()
    tddfa = TDDFA()

    tic = time.time()
    bboxes = face_boxes(img)
    toc = time.time()
    print(f'use {toc - tic}s to detect face')
    if len(bboxes) == 0 or bboxes is None:
        print('no face detected.')
        exit(0)

    bound_box = bboxes[np.argmax(bboxes[:,4])]
    # calculate Euler angle
    tic = time.time()
    param_lst, roi_box_lst = tddfa(img, [bound_box])
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)
    toc = time.time()
    print(f'use {toc - tic}s to get 3d face landmarks')

    landmarks = ver_lst[0]
    tic = time.time()
    rotation, landmarks = estimate_head_pose(landmarks, True)
    toc = time.time()
    print(f'use {toc - tic}s to estimate pose')
    print('pose: ', rotation)
    print('label: ', label)

    direction = get_direction_from_landmarks(landmarks)

    landmarks -= landmarks[30]
    landmarks = landmarks.tolist()

    arrows = []
    direction_h = direction[0]
    direction_v = direction[1]
    direction_d = direction[2]

    r_label = Rotation.from_euler(
        'ZYX', label, True
    ).apply(
        np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
    )

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
        },
        {
            "position": landmarks[30],
            "direction": r_label[0].tolist()
        },
        {
            "position": landmarks[30],
            "direction": r_label[1].tolist()
        },
        {
            "position": landmarks[30],
            "direction": r_label[2].tolist()
        }
    ]

    plot_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '3DPlot'
    )
    save_data_into_js(
        landmarks,
        arrows,
        os.path.join(plot_dir, 'js', 'data.js')
    )
    webbrowser.open(os.path.join(plot_dir, 'index.html'))
