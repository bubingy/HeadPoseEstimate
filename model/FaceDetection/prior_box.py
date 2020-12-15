# coding: utf-8

from itertools import product as product

import numpy as np

from model.FaceDetection.config import cfg


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.image_size = image_size
        self.feature_maps = [[np.ceil(self.image_size[0] / step), np.ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(int(f[0])), range(int(f[1]))):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in
                                    [j, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in
                                    [i, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in 
                        [j, j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in 
                        [i, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.reshape(np.array(anchors), (-1, 4)) 
        return output
