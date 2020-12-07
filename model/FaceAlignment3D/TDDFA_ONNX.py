# coding: utf-8

__author__ = 'cleardusk'

import os
import pickle
import numpy as np
import onnxruntime

from model.FaceAlignment3D.onnx import convert_to_onnx
from model.FaceAlignment3D.tddfa_util import _parse_param, similar_transform, parse_roi_box_from_bbox
from model.FaceAlignment3D.bfm import BFMModel
from model.FaceAlignment3D.bfm_onnx import convert_bfm_to_onnx


SCRIPT_HOME = os.path.dirname(os.path.abspath(__file__))


class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # torch.set_grad_enabled(False)

        # load onnx version of BFM
        bfm_fp = os.path.join(
            SCRIPT_HOME, 
            'weights', 
            'bfm_noneck_v3.pkl'
        )
        bfm_onnx_fp = bfm_fp.replace('.pkl', '.onnx')
        if not os.path.exists(bfm_onnx_fp):
            convert_bfm_to_onnx(
                bfm_onnx_fp,
                40,
                10
            )
        self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_fp, None)

        # load for optimization
        bfm = BFMModel(bfm_fp, shape_dim=40, exp_dim=10)
        self.tri = bfm.tri
        self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base

        # config
        self.size = 120

        param_mean_std_fp = os.path.join(
            SCRIPT_HOME,
            'weights',
            'param_mean_std_62d_120x120.pkl'
        )

        onnx_fp = os.path.join(
            SCRIPT_HOME, 
            'weights', 
            'mb1_120x120.pth'
        ).replace('.pth', '.onnx')

        # convert to onnx online if not existed
        if onnx_fp is None or not os.path.exists(onnx_fp):
            print(f'{onnx_fp} does not exist, try to convert the `.pth` version to `.onnx` online')
            onnx_fp = convert_to_onnx(**kvs)

        self.session = onnxruntime.InferenceSession(onnx_fp, None)

        # params normalization config
        r = pickle.load(open(param_mean_std_fp, 'rb'))
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, objs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        for obj in objs:
            roi_box = parse_roi_box_from_bbox(obj)

            roi_box_lst.append(roi_box)
            img = img_ori.crop(roi_box)
            img = img.resize((self.size, self.size))
            img = np.array(img)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            param = self.session.run(None, inp_dct)[0]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst):
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset
            pts3d = similar_transform(pts3d, roi_box, size)
            pts3d[1] *= -1
            ver_lst.append(np.transpose(pts3d))

        return ver_lst
