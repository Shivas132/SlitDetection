"""Module to search and test methods to find slits grow by the differences between frames."""

from image_process_utils import *
import numpy as np
import cv2 as cv

video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\outputs\exp_0_denoised.dat"

h = 3
template_window_size = 7
search_window_size = 21

data = frames_as_matrix_from_binary_file(video_path, offset=False)
# data = normalize_to_int(data, MAX_GRAY_VAL)

# --- create delta images for each frame: ---
frames_num, rows, cols = data.shape
deltas = np.empty((frames_num, rows, cols))
sum_of_deltas = np.zeros((rows, cols), np.float64)

for i in range(1, frames_num):
    delta_img = data[i] - data[i-1]
    delta_img = thresholding(delta_img, 0.05)
    sum_of_deltas = cv.bitwise_or(sum_of_deltas, delta_img)
    deltas[i] = sum_of_deltas


save_video(deltas, "exp_0_deltas_or")


# --- difference by flat field: ---

# flat_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\inputs\exp_1\flat_field.dat"
# flat = frames_as_matrix_from_binary_file(flat_path)
#
# diff = data - flat
# save_video(diff, "diff_video_exp01_flat")

