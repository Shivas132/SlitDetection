"""Module to search and test methods to find slits grow by the differences between frames."""

from image_process_utils import *
import numpy as np
import cv2 as cv
from paths import OUTPUTS

video_path = OUTPUTS+"exp_0_denoised.dat"

h = 3
template_window_size = 7
search_window_size = 21

data = frames_as_matrix_from_binary_file(video_path, offset=False)


# --- create delta images for each frame: ---
def deltas_video(data: np.array, thresh: float = 0.02) -> np.array:
    frames_num, rows, cols = data.shape
    deltas = np.empty((frames_num, rows, cols), dtype=np.uint8)
    sum_of_deltas = np.zeros((rows, cols), np.float64)
    for i in range(1, frames_num):
        delta_img = data[i] - data[i-1]
        delta_img = thresholding(delta_img, thresh)
        sum_of_deltas = cv.bitwise_or(sum_of_deltas, delta_img)
        deltas[i] = sum_of_deltas
    return deltas


deltas = deltas_video(data)
save_video(deltas, "exp_0_deltas_or_thresh0.02")

