# coding=utf-8

import cv2 as cv
import numpy as np

COLOR = {
    'red':   (255, 0,   0),
    'green': (0,   255, 0),
    'blue':  (0,   0, 255)
}


def draw_pose(img_, directions_lst, bound_box_lst, landmarks_lst, show_bbox=False, show_landmarks=False):
    img = img_.copy()

    if show_bbox:
        for bound_box in bound_box_lst:
            x_min,y_min,x_max,y_max = bound_box[:4]
            x_min,y_min,x_max,y_max = int(x_min),int(y_min),int(x_max),int(y_max)
            cv.rectangle(img, (x_min,y_min), (x_max,y_max), COLOR['green'], 1)

    if show_landmarks:
        for landmarks in landmarks_lst:
            for point in landmarks[:, :2]:
                cv.circle(
                    img, 
                    tuple(np.abs(point).astype(int)),
                    1, 
                    COLOR['green'], 
                    -1
                )

    for bound_box, directions in zip(bound_box_lst, directions_lst):
        tdx, tdy = (bound_box[:2]+bound_box[2:4])/2
        size = bound_box[2] - bound_box[0]
        # X-Axis drawn in red
        x1 = size * directions[0][0] + tdx
        y1 = -size * directions[0][1] + tdy

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


def plot_image(img, win_size, win_name='pose'):
    cv.imshow(win_name, cv.resize(img, win_size))
    cv.waitKey()
