"""Utility functions for image processing"""

from matplotlib import pyplot as plt
import numpy as np


OFFSET = 6336
FRAMES_NUM = 128
FRAME_HEIGHT = 250
FRAME_WIDTH = 400


def frames_as_matrix_from_binary_file(video_file_path):
    # Open the .dat file in binary mode
    with open(video_file_path, 'rb') as f:
        # Move the file pointer to the specified offset
        f.seek(OFFSET)
        # Read the data into a NumPy array
        data = np.fromfile(f, dtype='<i2', count=FRAMES_NUM * FRAME_HEIGHT * FRAME_WIDTH)

    # Reshape the array into the desired dimensions
    data = data.reshape((FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH))
    # Normalize the array to [0,1] grayscale
    data_norm = np.float64((data * (1 / np.max(data))))
    return data_norm


def show_frame(img, title='', figsize=(20, 16)):
    img = img * (255 / np.max(img))
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

