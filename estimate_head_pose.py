# coding=utf-8

import os
import json
import argparse
import webbrowser

import numpy as np
import torch
from PIL import Image

from model.face_alignment import FaceAlignment
from model.mtcnn import MTCNN
from model.deep_face import detect_face, estimate_head_pose, \
    get_direction_from_landmarks
from utils.utils import save_data_into_js


if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    parser = argparse.ArgumentParser(description='estimate head pose.')
    parser.add_argument('-i', '--image-path', required=True,
                        help="path of image.")
    args = parser.parse_args()
    img_path = args.image_path
    img = Image.open(img_path)

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
    webbrowser.open(os.path.join(plot_dir, 'index.html'))
