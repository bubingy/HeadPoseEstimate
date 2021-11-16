# coding=utf-8

import os
import argparse

import torch

from model.pose import estimate_head_pose
from model.plot import cv, draw_pose


if __name__ == "__main__":
    torch.set_grad_enabled(False) # disable auto grad
    parser = argparse.ArgumentParser(description='estimate head pose.')
    parser.add_argument(
        '-i', '--input-video',
        default='./figures/TheHill.gif',
        help="path of input video."
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

    video_path = args.input_video
    cap = cv.VideoCapture(video_path)

    fps = cap.get(cv.CAP_PROP_FPS)
    prefix_path, suffix_path = os.path.splitext(video_path)
    output_path = prefix_path + '_out.mp4'
    out = cv.VideoWriter(
        output_path,
        cv.VideoWriter_fourcc(*'XVID'),
        int(fps),
        (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
    )
    frames = []
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        bboxes = face_boxes(img)
        if len(bboxes) == 0 or bboxes is None:
            print('no face detected.')
            exit(0)

        # calculate Euler angle
        param_lst, roi_box_lst = tddfa(img, bboxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)

        euler_angle_lst, directions_lst, landmarks_lst = estimate_head_pose(
            ver_lst, True
        )
        show_img = draw_pose(
            img,
            directions_lst,
            bboxes,
            landmarks_lst,
            show_bbox=args.show_boundbox,
            show_landmarks=args.show_landmarks
        )
        out.write(show_img)
        frames.append(show_img)
        cv.imshow('pose', show_img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv.destroyAllWindows()
