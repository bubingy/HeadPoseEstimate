# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn


class BFMModel_ONNX(nn.Module):
    """BFM serves as a decoder"""

    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        super(BFMModel_ONNX, self).__init__()

        # load bfm
        bfm = pickle.load(open(bfm_fp, 'rb'))

        u = torch.from_numpy(bfm.get('u').astype(np.float32))
        self.u = u.view(-1, 3).transpose(1, 0)
        w_shp = torch.from_numpy(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        w_exp = torch.from_numpy(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        w = torch.cat((w_shp, w_exp), dim=1)
        self.w = w.view(-1, 3, w.shape[-1]).contiguous().permute(1, 0, 2)

    def forward(self, *inps):
        R, offset, alpha_shp, alpha_exp = inps
        alpha = torch.cat((alpha_shp, alpha_exp))
        pts3d = R @ (self.u + self.w.matmul(alpha).squeeze()) + offset
        return pts3d


def convert_bfm_to_onnx(bfm_onnx_fp, shape_dim=40, exp_dim=10):
    # print(shape_dim, exp_dim)
    bfm_fp = bfm_onnx_fp.replace('.onnx', '.pkl')
    bfm_decoder = BFMModel_ONNX(bfm_fp=bfm_fp, shape_dim=shape_dim, exp_dim=exp_dim)
    bfm_decoder.eval()

    dummy_input = torch.randn(3, 3), torch.randn(3, 1), torch.randn(shape_dim, 1), torch.randn(exp_dim, 1)
    R, offset, alpha_shp, alpha_exp = dummy_input
    torch.onnx.export(
        bfm_decoder,
        (R, offset, alpha_shp, alpha_exp),
        bfm_onnx_fp,
        input_names=['R', 'offset', 'alpha_shp', 'alpha_exp'],
        output_names=['output'],
        dynamic_axes={
            'alpha_shp': [0],
            'alpha_exp': [0],
        },
        do_constant_folding=True
    )
    print(f'Convert {bfm_fp} to {bfm_onnx_fp} done.')
