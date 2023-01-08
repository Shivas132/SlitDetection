"""Module to search and test methods to find slits grow by the differences between frames."""

from image_process_utils import *
import numpy as np

video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\
                    dat_files\outputs\denoised_video_exp.dat"

h = 3
template_window_size = 7
search_window_size = 21

with open(video_path, 'rb') as f:
    data = np.fromfile(f, dtype='<i2', count=128 * FRAME_HEIGHT * FRAME_WIDTH)
data = data.reshape((128, FRAME_HEIGHT, FRAME_WIDTH))

# --- create delta images for each frame: ---
frames_num, rows, cols = data.shape
deltas = np.empty((frames_num, rows, cols))
sum_of_deltas = np.zeros((rows, cols))

for i in range(1, frames_num):
    delta_img = data[i] - data[i-1]
    sum_of_deltas = sum_of_deltas + delta_img
    sum_of_deltas = thresholding(sum_of_deltas, 100)
    deltas[i] = sum_of_deltas

save_video(deltas, "deltas_video")
