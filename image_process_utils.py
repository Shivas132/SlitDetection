"""Utility functions for image processing"""

from matplotlib import pyplot as plt
import numpy as np


OFFSET = 6336
FRAMES_NUM = 128
FRAME_HEIGHT = 250
FRAME_WIDTH = 400
MAX_GRAY_VAL = 255


def normalize_to_float(data: np.array) -> np.array:
    return np.float64(data) / np.max(data)


def normalize_to_int(data: np.array, max_val: int = MAX_GRAY_VAL) -> np.array:
    return np.uint8(data * (max_val / np.max(data)))


def frames_as_matrix_from_binary_file(video_file_path, offset=True):
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
    img = normalize_to_int(img, MAX_GRAY_VAL)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def save_video(video, name):
    out_file = open(
        rf"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\outputs\{name}.dat", 'wb')
    video.astype('int16').tofile(out_file, sep='', format='%d')
    out_file.close()


def compare_2_frames(src, denoised, title1='Original', title2='denoised'):
    plt.figure(figsize=(20, 16))
    origin = plt.subplot(211)
    plt.imshow(src, cmap='gray')
    origin.set_title(title1)

    denoise_img = plt.subplot(212)
    plt.imshow(denoised, cmap='gray')
    denoise_img.set_title(title2)
    # denoise_img.set_title(f"denoise: h={h}, search window={search_window_size}, template window={template_window_size}")
    plt.show()


def thresholding(img: np.array, thresh: float):
    out_img = np.zeros(img.shape)
    out_img[img >= thresh] = 1
    return out_img
