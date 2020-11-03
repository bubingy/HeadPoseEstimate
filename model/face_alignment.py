# coding=utf-8

import os
import math

import torch
from enum import Enum
import numpy as np
from PIL import Image

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from model.FAN import FAN, ResNetDepth


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = np.asarray(newImg)
    newImg = Image.fromarray(newImg)
    newImg = newImg.resize((int(resolution), int(resolution)))
    return np.array(newImg)


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


class FaceAlignment:
    def __init__(self, fan_weight, depthnet_weight, device='cpu'):
        self.fan_weight = fan_weight
        self.depthnet_weight = depthnet_weight
        self.device = device

        network_size = 4

        self.reference_scale = 195

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)

        self.face_alignment_net.load_state_dict(
            torch.load(self.fan_weight, map_location=self.device)
        )

        self.face_alignment_net.to(self.device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        self.depth_prediciton_net = ResNetDepth()

        depth_weights = torch.load(self.depthnet_weight, map_location=self.device)
        depth_dict = {
            k.replace('module.', ''): v for k,
            v in depth_weights['state_dict'].items()}
        self.depth_prediciton_net.load_state_dict(depth_dict)

        self.depth_prediciton_net.to(self.device)
        self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_face):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_face {numpy.array} -- bounding boxes for face found
            in the image (default: {None})
        """
        return self.get_landmarks_from_image(image_or_path, detected_face)

    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_face):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_face {numpy.array} -- bounding box for face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = np.array(Image.open(image_or_path))
            except IOError:
                print("error opening file : ", image_or_path)
                return None
        elif isinstance(image_or_path, torch.Tensor):
            image = image_or_path.detach().cpu().numpy()
        else:
            image = image_or_path

        assert image.ndim == 3

        d = detected_face
        center = torch.FloatTensor(
            [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
        center[1] = center[1] - (d[3] - d[1]) * 0.12
        scale = (d[2] - d[0] + d[3] - d[1]) / self.reference_scale

        inp = crop(image, center, scale)
        inp = torch.from_numpy(inp.transpose(
            (2, 0, 1))).float()

        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)

        out = self.face_alignment_net(inp)[-1].detach()
        out = out.cpu()

        pts, pts_img = get_preds_fromhm(out, center, scale)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

        heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
        for i in range(68):
            if pts[i, 0] > 0:
                heatmaps[i] = draw_gaussian(
                    heatmaps[i], pts[i], 2)
        heatmaps = torch.from_numpy(
            heatmaps).unsqueeze_(0)

        heatmaps = heatmaps.to(self.device)
        depth_pred = self.depth_prediciton_net(
            torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
        pts_img = torch.cat(
            (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

        return pts_img.numpy()