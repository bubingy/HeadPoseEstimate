# coding: utf-8

__author__ = 'cleardusk'

import os
import pickle

import numpy as np
import torch
from torchvision.transforms import Compose

from model.FaceAlignment3D.bfm import BFMModel
from model.FaceAlignment3D.mobilenet_v1 import mobilenet
from model.FaceAlignment3D.tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz, parse_roi_box_from_bbox
)


SCRIPT_HOME = os.path.dirname(os.path.abspath(__file__))


class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self):
        torch.set_grad_enabled(False)

        # load BFM
        self.bfm = BFMModel(
            bfm_fp=os.path.join(
                SCRIPT_HOME, 
                'weights', 
                'bfm_noneck_v3.pkl'
            ),
            shape_dim=40,
            exp_dim=10
        )
        self.tri = self.bfm.tri

        # config
        self.size = 120

        param_mean_std_fp = os.path.join(
            SCRIPT_HOME, 
            'weights', 
            'param_mean_std_62d_120x120.pkl'
        )

        # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
        model = mobilenet(
            num_classes=62,
            widen_factor=1
        )
        model = load_model(
            model,
            os.path.join(
                SCRIPT_HOME, 
                'weights', 
                'mb1_120x120.pth'
            )
        )

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = pickle.load(open(param_mean_std_fp, 'rb'))
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def __call__(self, img_ori, objs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: the list of box
        :return: param list and roi_box list
        """
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        for obj in objs:
            roi_box = parse_roi_box_from_bbox(obj)
            roi_box_lst.append(roi_box)
            img = img_ori.crop(roi_box)
            img = img.resize((self.size, self.size))
            img = np.array(img)
            inp = self.transform(img).unsqueeze(0)

            param = self.model(inp)

            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst):
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            pts3d = R @ (self.bfm.u_base + \
                    self.bfm.w_shp_base @ alpha_shp + \
                    self.bfm.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset
            pts3d = similar_transform(pts3d, roi_box, size)
            pts3d[1] *= -1
            ver_lst.append(np.transpose(pts3d))

        return ver_lst
