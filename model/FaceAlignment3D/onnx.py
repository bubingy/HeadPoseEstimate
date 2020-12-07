# coding: utf-8

__author__ = 'cleardusk'

import os

import torch
import model.FaceAlignment3D as models
from model.FaceAlignment3D.tddfa_util import load_model

SCRIPT_HOME = os.path.dirname(os.path.abspath(__file__))

def convert_to_onnx():
    # 1. load model
    size = 120
    model = getattr(
        models,
        'mobilenet'
    )(
        num_classes=62,
        widen_factor=1,
        size=size,
        mode='small'
    )
    checkpoint_fp = os.path.join(
        SCRIPT_HOME, 
        'weights', 
        'mb1_120x120.pth'
    )
    model = load_model(model, checkpoint_fp)
    model.eval()

    # 2. convert
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, size, size)
    wfp = checkpoint_fp.replace('.pth', '.onnx')
    torch.onnx.export(
        model,
        (dummy_input, ),
        wfp,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True
    )
    print(f'Convert {checkpoint_fp} to {wfp} done.')
    return wfp
