"""Module to search and test denoising methods."""
import numpy as np

from image_process_utils import *
from matplotlib import pyplot as plt
import cv2 as cv


video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\inputs\exp_0\exp.dat"
data = frames_as_matrix_from_binary_file(video_path)
show_frame(data[64])

data = normalize_to_int(data, MAX_GRAY_VAL)
show_frame(data[64])
# select a frame with a slit:
frame_num = 64
img_to_denoise = data[frame_num]

# --------------------- non-local means denoising: --------------------------
h = 3
template_window_size = 7
search_window_size = 21

# --- creating denoised video: ---
new_video = np.empty(data.shape)

for i in range(data.shape[0]):
    new_video[i] = cv.fastNlMeansDenoising(data[i], h=h, templateWindowSize=template_window_size,
                                           searchWindowSize=search_window_size)


save_video(new_video, "exp_0_denoised")

# background_frame = cv.fastNlMeansDenoising(data[0], h=h, templateWindowSize=template_window_size,
#                                            searchWindowSize=search_window_size)
#
# denoise_data = cv.fastNlMeansDenoising(img_to_denoise, h=h, templateWindowSize=template_window_size,
#                                        searchWindowSize=search_window_size)
#
# plt.figure(figsize=(20, 16))
# origin = plt.subplot(211)
# plt.imshow(img_to_denoise, cmap='gray')
# origin.set_title("Original")
#
# denoise_img = plt.subplot(212)
# plt.imshow(denoise_data, cmap='gray')
# denoise_img.set_title(f"denoise: h={h}, search window={search_window_size}, template window={template_window_size}")
# plt.show()

# # --- Detect edges with Sobel: ---
# sobel_x = cv.Sobel(src=denoise_data, ddepth=cv.CV_8U, dx=1, dy=0, ksize=5)
# sobel_y = cv.Sobel(src=denoise_data, ddepth=cv.CV_8U, dx=0, dy=1, ksize=5)
# sobel_xy = cv.Sobel(src=denoise_data, ddepth=cv.CV_8U, dx=1, dy=1, ksize=5)
#
# plt.figure(figsize=(20, 16))
# x_edges = plt.subplot(311)
# plt.imshow(sobel_x, cmap='gray')
# x_edges.set_title('sobel x edges')
#
# y_edges = plt.subplot(312)
# plt.imshow(sobel_y, cmap='gray')
# y_edges.set_title('sobel y edges')
#
# xy_edges = plt.subplot(313)
# plt.imshow(sobel_xy, cmap='gray')
# xy_edges.set_title('sobel xy edges')

# plt.show()


# --------------------- bilateral denoising: --------------------------

# pix_hood = 5
# sigma_color = 50
# sigma_space = 50
#
# bilaterl = cv.bilateralFilter(img_to_denoise, pix_hood, sigma_color, sigma_space)
# plt.figure(figsize=(20, 16))
# origin = plt.subplot(211)
# plt.imshow(img_to_denoise, cmap='gray')
# origin.set_title("Original")
#
# denoise_img = plt.subplot(212)
# plt.imshow(bilaterl, cmap='gray')
# denoise_img.set_title(f"bilaterl denoise - pixel neighbourhood={pix_hood}, sigma color={sigma_color},"
#                       f" sigma space={sigma_space}")
# plt.show()
