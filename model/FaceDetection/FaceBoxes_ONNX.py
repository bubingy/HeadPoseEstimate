# coding: utf-8

import time
import os.path as osp

import numpy as np
from PIL import Image
import onnxruntime

from model.FaceDetection.prior_box import PriorBox
from model.FaceDetection.faceboxes_utils import cpu_nms, decode
from model.FaceDetection.config import cfg
from model.FaceDetection.onnx import convert_to_onnx


# some global configs
confidence_threshold = 0.05
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
onnx_path = make_abs_path('weights/FaceBoxesProd.onnx')


class FaceBoxes_ONNX(object):
    def __init__(self):
        if not osp.exists(onnx_path):
            convert_to_onnx(onnx_path)
        self.session = onnxruntime.InferenceSession(onnx_path, None)

    def __call__(self, img_):
        img_ = np.array(img_)
        img_raw = img_.copy()

        # scaling to speed up
        img = np.float32(img_raw)

        # forward
        im_height, im_width, _ = img.shape
        scale_bbox = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]

        loc, conf = self.session.run(None, {'input': img})

        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        boxes = decode(np.squeeze(loc, axis=0), priors, cfg['variance'])
        boxes = boxes * scale_bbox

        scores = conf[0][:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]

        # filter using vis_thres
        det_bboxes = []
        for b in dets:
            if b[4] > vis_thres:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return np.array(det_bboxes)
