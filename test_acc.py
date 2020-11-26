# coding=utf-8

import os
from threading import Thread, Lock
from multiprocessing import cpu_count

import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

from model.mtcnn import MTCNN
from model.face_alignment import FaceAlignment
from model.deep_face import detect_face, estimate_head_pose
from utils.utils import data_loader, get_num_of_images, \
    get_euler_angles_from_rotation_matrix, get_mat_from_txt
from utils.global_vars import *


def calculate_error():
    global roll_error_list, yaw_error_list, pitch_error_list, mutex, data, num_images

    mtcnn = MTCNN(image_size=224, device=torch.device('cpu'))
    face_alighment = FaceAlignment('model/3DFAN4.pth.tar', 'model/depth.pth.tar')

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

        label = get_euler_angles_from_rotation_matrix(
            get_mat_from_txt(label_path)
        )

        img = Image.open(img_path)

        bound_box = None
        try:
            bound_box = detect_face(img, mtcnn)
        except Exception:
            continue

        # calculate Euler angle
        predict = estimate_head_pose(img, bound_box, face_alighment)

        err = np.abs(np.abs(predict) - np.abs(label))
        # if max(err) > 100:
        #     print(img_path)
        #     print('label: ', label)
        #     print('predict: ', predict)
        #     input()

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
