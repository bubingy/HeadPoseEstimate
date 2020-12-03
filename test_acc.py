"""Test against Biwi Kinect Head Pose Database"""
# coding=utf-8

from threading import Thread, Lock
from multiprocessing import cpu_count

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model.FaceDetection.FaceBoxes import FaceBoxes
from model.FaceAlignment3D.TDDFA import TDDFA
from model.pose import estimate_head_pose, get_direction_from_landmarks
from utils.utils import data_loader, get_num_of_images, \
    get_euler_angles_from_rotation_matrix, get_mat_from_txt
from utils.global_vars import *


def calculate_error():
    global roll_error_list, yaw_error_list, pitch_error_list, mutex, data, num_images

    face_boxes = FaceBoxes()
    tddfa = TDDFA()

    while True:
        img_path = ''
        label_path = ''
        mutex.acquire()
        try:
            img_path, label_path = next(data)
        except StopIteration:
            mutex.release()
            return
        mutex.release()

        img = Image.open(img_path)
        label = get_euler_angles_from_rotation_matrix(
            get_mat_from_txt(label_path)
        )

        bboxes = face_boxes(img)
        if len(bboxes) == 0:
            continue
        bound_box = bboxes[np.argmax(bboxes[:,4])]

        param_lst, roi_box_lst = tddfa(img, [bound_box])
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)
        landmarks = ver_lst[0]

        rotation = estimate_head_pose(landmarks)

        err = np.abs(np.abs(rotation) - np.abs(label))

        mutex.acquire()
        roll_error_list.append([label[0], err[0]])
        yaw_error_list.append([label[1], err[1]])
        pitch_error_list.append([label[2], err[2]])
        num_images -= 1
        if num_images % 10 == 0:
            print(f'processing, has {num_images} left...')
        mutex.release()


def task_runner(root_directory: str):
    """Run all tasks util finished.

    Args:
        root_directory: directory where store images and label files.
    """
    global num_images
    num_tasks = min(cpu_count(), num_images)
    tasks_list = []
    for _ in range(num_tasks):
        task = Thread(target=calculate_error)
        task.start()
        tasks_list.append(task)
    for task in tasks_list:
        task.join()


if __name__ == "__main__":
    root_directory = 'F:\\DATA\\Biwi_Kinect_Head_Pose_Database\\03'

    torch.set_grad_enabled(False)

    mutex = Lock()
    data = data_loader(root_directory)
    num_images = get_num_of_images(root_directory)

    task_runner(root_directory)
    
    roll_error_list.sort(key=lambda item: item[0])
    yaw_error_list.sort(key=lambda item: item[0])
    pitch_error_list.sort(key=lambda item: item[0])

    roll_error_list = np.array(roll_error_list)
    yaw_error_list = np.array(yaw_error_list)
    pitch_error_list = np.array(pitch_error_list)

    np.save('roll_err', roll_error_list)
    np.save('yaw_err', yaw_error_list)
    np.save('pitch_err', pitch_error_list)

    plt.subplot(311)
    plt.title('error: roll')
    plt.plot(roll_error_list[:,0], roll_error_list[:,1])
    plt.subplot(312)
    plt.title('error: yaw')
    plt.plot(yaw_error_list[:,0], yaw_error_list[:,1])
    plt.subplot(313)
    plt.title('error: pitch')
    plt.plot(pitch_error_list[:,0], pitch_error_list[:,1])

    plt.show()
