"""Module to search and test methods to find slits grow by the differences between frames."""

from image_process_utils import *
import numpy as np
import cv2 as cv


# --- create delta images for each frame: ---
def deltas_video(data: np.array, thresh: float = 0.02) -> np.array:
    """
        Calculates delta frames for a given video data.

        Args:
            data (numpy.ndarray): The input video data.
            thresh (float, optional): The threshold for determining changes between frames. Defaults to 0.02.

        Returns:
            numpy.ndarray: The delta frames.
    """
    frames_num, rows, cols = data.shape
    deltas = np.empty((frames_num, rows, cols), dtype=np.uint8)
    sum_of_deltas = np.zeros((rows, cols), np.float64)
    for i in range(1, frames_num):
        delta_img = data[i] - data[i-1]
        delta_img = thresholding(delta_img, thresh)
        sum_of_deltas = cv.bitwise_or(sum_of_deltas, delta_img)
        deltas[i] = sum_of_deltas
    return deltas
