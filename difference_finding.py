"""Module to search and test methods to find slits grow by the differences between frames."""

from image_process_utils import frames_as_matrix_from_binary_file, show_frame
import cv2 as cv

video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\CAM1_22_41_46.dat"
data = frames_as_matrix_from_binary_file(video_path)

# select a frame with a slit:
frame_num = 64
img = data[frame_num]

# --- using SSD between background frame and each frame: ---

method = 'cv2.TM_SQDIFF'