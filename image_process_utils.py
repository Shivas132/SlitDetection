"""Utility functions for image processing"""

from matplotlib import pyplot as plt
import numpy as np
from paths import *


OFFSET = 6336
FRAMES_NUM = 128
FRAME_HEIGHT = 250
FRAME_WIDTH = 400
MAX_GRAY_VAL = 255


def normalize_to_float(data: np.array) -> np.array:
    """
        Normalizes an array to the range [0, 1] by converting it to float.

        Args:
            data (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The normalized array.
    """
    return np.float64(data) / np.max(data)


def normalize_to_int(data: np.array, max_val: int = MAX_GRAY_VAL) -> np.array:
    """
        Normalizes an array to the range [0, max_val] by converting it to unsigned integer.

        Args:
            data (numpy.ndarray): The input array.
            max_val (int, optional): The maximum value for normalization. Defaults to MAX_GRAY_VAL.

        Returns:
            numpy.ndarray: The normalized array.
    """
    return np.uint8(data * (max_val / np.max(data)))


def frames_as_matrix_from_binary_file(video_file_path, offset=True):
    """
        Reads frames from a binary file and returns them as a matrix.

        Args:
            video_file_path (str): The path to the binary file.
            offset (bool, optional): Whether to apply an offset while reading the file. Defaults to True.

        Returns:
            numpy.ndarray: The frames as a matrix.
    """
    # Open the .dat file in binary mode
    with open(video_file_path, 'rb') as f:
        if offset:
            # Move the file pointer to the specified offset
            f.seek(OFFSET)
        # Read the data into a NumPy array
        data = np.fromfile(f, dtype='<i2', count=FRAMES_NUM * FRAME_HEIGHT * FRAME_WIDTH)

    # Reshape the array into the desired dimensions
    data = data.reshape((FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH))
    # Normalize the array to [0,1] grayscale
    data = normalize_to_float(data)
    return data


def show_frame(img, title='', figsize=(20, 16)):
    """
        Displays an image.

        Args:
            img (numpy.ndarray): The image to display.
            title (str, optional): The title of the image. Defaults to ''.
            figsize (tuple, optional): The figure size. Defaults to (20, 16).

        Returns:
            None
    """
    img = normalize_to_int(img, MAX_GRAY_VAL)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def save_video(video, name, directory=OUTPUTS):
    """
        Saves a video to a binary file.

        Args:
            video (numpy.ndarray): The video to save.
            name (str): The name of the file.
            directory (str, optional): The directory to save the file. Defaults to OUTPUTS.

        Returns:
            None
    """
    out_file = open(
        f"{directory}{name}.dat", 'wb')
    video.astype('int16').tofile(out_file, sep='', format='%d')
    out_file.close()


def compare_2_frames(img1, img2, title1='img1', title2='img2'):
    """
        Compares and displays two frames side by side.

        Args:
            img1 (numpy.ndarray): The first frame.
            img2 (numpy.ndarray): The second frame.
            title1 (str, optional): The title of the first frame. Defaults to 'img1'.
            title2 (str, optional): The title of the second frame. Defaults to 'img2'.

        Returns:
            None
    """
    plt.figure(figsize=(20, 16))
    origin = plt.subplot(211)
    plt.imshow(img1, cmap='gray')
    origin.set_title(title1)

    denoise_img = plt.subplot(212)
    plt.imshow(img2, cmap='gray')
    denoise_img.set_title(title2)
    plt.show()


def thresholding(img: np.array, thresh: float):
    out_img = np.zeros(img.shape)
    out_img[img >= thresh] = 1
    return out_img
