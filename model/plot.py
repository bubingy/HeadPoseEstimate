# coding=utf-8

import cv2 as cv
import numpy as np

COLOR = {
    'red':   (255, 0,   0),
    'green': (0,   255, 0),
    'blue':  (0,   0, 255)
}

def draw_landmarks(img_, pts, color=(0, 255, 0), size=2):
    img = img_.copy()

    for pt in pts:
        img = cv.circle(img, (int(pt[0]), -int(pt[1])), size, color, -1)

    return img

def draw_pose(img_, directions, centroid, bound_box=None, landmarks=None):
    img = img_.copy()

    if bound_box is None:
        size = 60
    else:
        size = bound_box[2] - bound_box[0]
        x_min,y_min,x_max,y_max = bound_box[:4]
        x_min,y_min,x_max,y_max = int(x_min),int(y_min),int(x_max),int(y_max)
        img = cv.rectangle(img, (x_min,y_min), (x_max,y_max), COLOR['green'], 1)

    if landmarks is not None:
        for pt in landmarks:
            img = cv.circle(img, (int(pt[0]), -int(pt[1])), 1, COLOR['green'], -1)

    tdx, tdy = np.abs(centroid[:2])

    # X-Axis drawn in red
    x1 = size * directions[0][0] + tdx
    y1 = size * directions[0][1] + tdy

    # Y-Axis drawn in green
    x2 = -size * directions[1][0] + tdx
    y2 = size * directions[1][1] + tdy

    # Z-Axis drawn in blue
    x3 = size * directions[2][0] + tdx
    y3 = -size * directions[2][1] + tdy

    cv.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def plot_image(img, win_name='pose'):
    cv.imshow(win_name, img)
    cv.waitKey()
