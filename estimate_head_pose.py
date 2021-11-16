# coding=utf-8

import os
import time
import argparse

import torch

from model.pose import estimate_head_pose
from model.plot import cv, draw_pose, plot_image

if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    parser = argparse.ArgumentParser(description='estimate head pose.')
    parser.add_argument(
        '-i', '--input-image',
        default='./figures/origin_image.png',
        help="path of image."
    )
    parser.add_argument(
        '--show-boundbox',
        action='store_true',
        default=False,
        help="whether to show bound box of face."
    )
    parser.add_argument(
        '--show-landmarks',
        action='store_true',
        default=False,
        help="whether to show facial landmarks."
    )
    parser.add_argument(
        '--onnx',
        action='store_true',
        default=False,
        help="whether to run on onnx runtime."
    )
    args = parser.parse_args()

    img_path = args.input_image
    img = cv.imread(img_path)

    # initialize networks
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '1'
        from model.FaceDetection.FaceBoxes_ONNX import FaceBoxes_ONNX
        from model.FaceAlignment3D.TDDFA_ONNX import TDDFA_ONNX
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX()
    else:
        from model.FaceDetection.FaceBoxes import FaceBoxes
        from model.FaceAlignment3D.TDDFA import TDDFA
        face_boxes = FaceBoxes()
        tddfa = TDDFA()

    tic = time.time()
    bboxes = face_boxes(img)
    if len(bboxes) == 0 or bboxes is None:
        print('no face detected.')
        exit(0)
    print(f'{len(bboxes)} faces detected.')

    # calculate Euler angle
    param_lst, roi_box_lst = tddfa(img, bboxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)

    euler_angle_lst, directions_lst, landmarks_lst = estimate_head_pose(
        ver_lst, True
    )
    toc = time.time()
    for euler_angle in euler_angle_lst:
        roll, yaw, pitch = euler_angle
        print(f'roll: {roll}, yaw: {yaw}, pitch: {pitch}. cost time: {toc-tic}s')

    show_img = draw_pose(
        img,
        directions_lst,
        bboxes,
        landmarks_lst,
        show_bbox=args.show_boundbox,
        show_landmarks=args.show_landmarks
    )
    prefix_path, suffix_path = os.path.splitext(img_path)
    output_path = prefix_path + '_out' + suffix_path
    cv.imwrite(output_path, show_img)
    plot_image(show_img, (1440, 900))
