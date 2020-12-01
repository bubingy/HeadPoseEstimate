# coding=utf-8

import os
import json
import time
import argparse
import webbrowser

import numpy as np
import torch
from PIL import Image

from model.FaceDetection.mtcnn import MTCNN
from model.FaceAlignment3D.TDDFA import TDDFA
from model.deep_face import detect_face, estimate_head_pose, \
    get_direction_from_landmarks
from utils.utils import save_data_into_js


if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    parser = argparse.ArgumentParser(description='estimate head pose.')
    parser.add_argument(
        '-i', '--image-path', required=True,
        help="path of image."
    )
    args = parser.parse_args()

    img_path = args.image_path
    img = Image.open(img_path)

    # initialize three networks
    # mtcnn: face detection
    # tddfa: get 68 3d face landmarks
    mtcnn = MTCNN(device=torch.device('cpu'))
    tddfa = TDDFA()

    tic = time.time()
    bound_box = detect_face(img, mtcnn)
    toc = time.time()
    print(f'use {toc - tic}s to detect face')
    if bound_box is None:
        print('no face detected.')
        exit(0)

    # calculate Euler angle
    tic = time.time()
    rotation, landmarks = estimate_head_pose(img, bound_box, tddfa, True)
    toc = time.time()
    print(f'use {toc - tic}s to estimate pose')
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

    plot_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '3DPlot'
    )
    save_data_into_js(
        landmarks, 
        arrows,
        os.path.join(plot_dir, 'js', 'data.js')
    )
    # webbrowser.open(os.path.join(plot_dir, 'index.html'))
