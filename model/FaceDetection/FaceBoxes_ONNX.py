# coding: utf-8

import os.path as osp

import numpy as np
from PIL import Image
import onnxruntime

from .prior_box import PriorBox
from .faceboxes_utils import cpu_nms, decode
from .config import cfg
from .onnx import convert_to_onnx


# some global configs
confidence_threshold = 0.05
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5
resize = 1

scale_flag = True
HEIGHT, WIDTH = 720, 1080

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
        scale = 1
        if scale_flag:
            h, w = img_raw.shape[:2]
            if h > HEIGHT:
                scale = HEIGHT / h
            if w * scale > WIDTH:
                scale *= WIDTH / (w * scale)
            # print(scale)
            if scale == 1:
                img_raw_scale = img_raw
            else:
                h_s = int(scale * h)
                w_s = int(scale * w)
                img_raw_scale = Image.fromarray(img_raw).resize((w_s, h_s))


            img = np.float32(img_raw_scale)
        else:
            img = np.float32(img_raw)

        # forward
        im_height, im_width, _ = img.shape
        scale_bbox = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]

        # loc, conf = self.net(img)  # forward pass
        out = self.session.run(None, {'input': img})
        loc, conf = out[0], out[1]

        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        boxes = decode(np.squeeze(loc, axis=0), priors, cfg['variance'])
        if scale_flag:
            boxes = boxes * scale_bbox / scale / resize
        else:
            boxes = boxes * scale_bbox / resize

        scores = conf[0][:, 1]
        # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

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

        return det_bboxes

