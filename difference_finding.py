"""Module to search and test methods to find slits grow by the differences between frames."""

from image_process_utils import *
import numpy as np

video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\inputs\exp_1\exp.dat"

h = 3
template_window_size = 7
search_window_size = 21

# with open(video_path, 'rb') as f:
#     data = np.fromfile(f, dtype='<i2', count=128 * FRAME_HEIGHT * FRAME_WIDTH)
# data = data.reshape((FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH))

data = frames_as_matrix_from_binary_file(video_path)

# --- create delta images for each frame: ---
frames_num, rows, cols = data.shape
deltas = np.empty((frames_num, rows, cols))
sum_of_deltas = np.zeros((rows, cols))

for i in range(1, frames_num):
    delta_img = data[i] - data[i-1]
    # sum_of_deltas = thresholding(sum_of_deltas, 100)
    sum_of_deltas = sum_of_deltas + delta_img  # logic OR
    deltas[i] = sum_of_deltas

for i in range(frames_num):
    deltas[i] = thresholding(deltas[i], 0)

save_video(deltas, "exp_1_deltas_thresh0")


# --- difference by flat field: ---

# flat_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\inputs\exp_1\flat_field.dat"
# flat = frames_as_matrix_from_binary_file(flat_path)
#
# diff = data - flat
# save_video(diff, "diff_video_exp01_flat")

