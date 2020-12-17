# coding=utf-8

import os
import argparse

import numpy as np
import torch

from model.pose import estimate_head_pose
from model.plot import cv, draw_pose, plot_image

if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    parser = argparse.ArgumentParser(description='estimate head pose.')
    parser.add_argument(
        '-i', '--image-path',
        default='./figures/origin_image.png',
        help="path of image."
    )
    parser.add_argument(
        '--onnx', 
        action='store_true',
        default=True,
        help="whether to run on onnx runtime."
    )
    args = parser.parse_args()

    img_path = args.image_path
    img = cv.imread(img_path)

    # initialize three networks
    # mtcnn: face detection
    # tddfa: get 68 3d face landmarks
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        from model.FaceDetection.FaceBoxes_ONNX import FaceBoxes_ONNX
        from model.FaceAlignment3D.TDDFA_ONNX import TDDFA_ONNX
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX()
    else:
        from model.FaceDetection.FaceBoxes import FaceBoxes
        from model.FaceAlignment3D.TDDFA import TDDFA
        face_boxes = FaceBoxes()
        tddfa = TDDFA()

    bboxes = face_boxes(img)
    if len(bboxes) == 0 or bboxes is None:
        print('no face detected.')
        exit(0)

    bound_box = bboxes[np.argmax(bboxes[:,4])]
    # calculate Euler angle
    param_lst, roi_box_lst = tddfa(img, [bound_box])
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)

    landmarks = ver_lst[0].copy()
    euler_angle, directions, landmarks = estimate_head_pose(
        landmarks, True
    )
    print(euler_angle)

    show_img = draw_pose(
        img,
        directions,
        np.mean(landmarks, axis=0),
        bound_box=bound_box,
        landmarks=landmarks
    )
    plot_image(show_img)
