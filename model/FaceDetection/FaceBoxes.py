# coding: utf-8

import os.path as osp

import torch
import numpy as np
from PIL import Image

from model.FaceDetection.faceboxesnet import FaceBoxesNet
from model.FaceDetection.prior_box import PriorBox
from model.FaceDetection.faceboxes_utils import cpu_nms, decode, load_model
from model.FaceDetection.config import cfg


# some global configs
confidence_threshold = 0.05
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5

scale_flag = True
HEIGHT, WIDTH = 720, 1080

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
pretrained_path = make_abs_path('weights/FaceBoxesProd.pth')


class FaceBoxes:
    def __init__(self):
        torch.set_grad_enabled(False)

        net = FaceBoxesNet(phase='test', size=None, num_classes=2)  # initialize detector
        self.net = load_model(net, pretrained_path=pretrained_path, load_to_cpu=True)
        self.net.eval()

    def __call__(self, img_):
        img_ = np.array(img_)
        img_raw = img_.copy()

        # scaling to speed up
        img = np.float32(img_raw)

        # forward
        im_height, im_width, _ = np.shape(img)
        scale_bbox = np.array([im_width, im_height, im_width, im_height])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        loc, conf = self.net(img)  # forward pass
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        boxes = decode(loc.data.squeeze(0).numpy(), priors, cfg['variance'])
        boxes = boxes * scale_bbox

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

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
