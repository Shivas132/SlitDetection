import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure, data
import numpy as np
import pandas as pd
from skimage.morphology import closing, square
import cv2 as cv
from image_process_utils import *


def test():
    video_path = f"{DELTAS}exp_1_deltas_thresh_0.05_orientation_0.01_size_4_eccen_0.7_filter.dat"
    video = frames_as_matrix_from_binary_file(video_path, offset=False)
    frame_num = 64
    frame = video[frame_num, :, :]
    # show_frame(frame)
    m, n = frame.shape
    # specify the size of each cube
    cube_size = 10
    #
    # # calculate the number of cubes along each axis
    # num_cubes_x = frame.shape[1] // cube_size
    # num_cubes_y = frame.shape[0] // cube_size
    #
    # # reshape the array into cubes
    # cubes = frame.reshape((num_cubes_y, cube_size, num_cubes_x, cube_size)).swapaxes(1, 2)
    #
    # # print the size of each cube
    # for i in range(num_cubes_y):
    #     for j in range(num_cubes_x):
    #         print(f"Cube ({i},{j}): {cubes[i,j].shape}")
    stds = {}
    ratios = {}
    ratio_thresh = 0.3
    mask = np.ones((m, n))
    for i in range(0, m, cube_size // 5):
        for j in range(0, n, cube_size // 5):
            cube = frame[i: i + cube_size, j: j + cube_size]
            std = np.std(cube)
            ratio = np.sum(cube) / np.square(cube_size)
            stds[(i, j)] = std
            ratios[(i, j)] = ratio
            if ratio > ratio_thresh:
                mask[i: i + cube_size, j: j + cube_size] = 0
    masked_frame = np.logical_and(frame, mask)
    # show_frame(masked_frame)
    masked_video = np.logical_and(video, mask)
    masked_video = normalize_to_int(masked_video)
    save_video(masked_video, f'exp1_object_remove_ratio_{ratio_thresh}_overlapping_2pxls', OUTPUTS)


test()
