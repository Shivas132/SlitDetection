from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from skimage.measure import regionprops

OFFSET = 6336
FRAMES_NUM = 128
FRAME_HEIGHT = 250
FRAME_WIDTH = 400

video_file_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\CAM1_22_41_46.dat"

# Open the .dat file in binary mode
with open(video_file_path, 'rb') as f:
    # Move the file pointer to the specified offset
    f.seek(OFFSET)

    # Read the data into a NumPy array
    data = np.fromfile(f, dtype='<i2', count=FRAMES_NUM * FRAME_HEIGHT * FRAME_WIDTH)

# Reshape the array into the desired dimensions
data = data.reshape((FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH))

# Normalize the array to [0-255] grayscale
data_norm = np.uint8((data * (255 / np.max(data))))

# select a frame with a slit:
frame_num = 64
img_to_denoise = data_norm[frame_num]


# --------------------- non-local means denoising: --------------------------
h = 3
template_window_size = 7
search_window_size = 21

denoise_data = cv.fastNlMeansDenoising(img_to_denoise, h=h, templateWindowSize=template_window_size,
                                       searchWindowSize=search_window_size)

plt.figure(figsize=(20, 16))
origin = plt.subplot(211)
plt.imshow(img_to_denoise, cmap='gray')
origin.set_title("Original")

denoise_img = plt.subplot(212)
plt.imshow(denoise_data, cmap='gray')
denoise_img.set_title(f"denoise: h={h}, search window={search_window_size}, template window={template_window_size}")
plt.show()

# --- Detect edges with Sobel: ---
sobel_x = cv.Sobel(src=denoise_data, ddepth=cv.CV_8U, dx=1, dy=0, ksize=5)
sobel_y = cv.Sobel(src=denoise_data, ddepth=cv.CV_8U, dx=0, dy=1, ksize=5)
sobel_xy = cv.Sobel(src=denoise_data, ddepth=cv.CV_8U, dx=1, dy=1, ksize=5)

plt.figure(figsize=(20, 16))
x_edges = plt.subplot(311)
plt.imshow(sobel_x, cmap='gray')
x_edges.set_title('sobel x edges')

y_edges = plt.subplot(312)
plt.imshow(sobel_y, cmap='gray')
y_edges.set_title('sobel y edges')

xy_edges = plt.subplot(313)
plt.imshow(sobel_xy, cmap='gray')
xy_edges.set_title('sobel xy edges')

plt.show()

# for i in range(30, 104):
#     frame = data_norm[i]
#     denoise_frame = cv.fastNlMeansDenoising(frame, h=h, templateWindowSize=template_window_size,
#                                             searchWindowSize=search_window_size)
#     sobel_frame = cv.Sobel(src=denoise_frame, ddepth=cv.CV_8U, dx=0, dy=1, ksize=5)
#     plt.figure(figsize=(10, 8))
#     plt.imshow(sobel_frame, cmap='gray')
#     plt.title(f"frame: {i}")
#     plt.show()


# --- Detect edges with Canny: ---
# T1 = 150
# T2 = 150
# canny = cv.Canny(image=denoise_data, threshold1=T1, threshold2=T2)
# plt.figure(figsize=(20, 16))
# denoise_img = plt.subplot(211)
# plt.imshow(denoise_data, cmap='gray')
# denoise_img.set_title("Original")
#
# canny_img = plt.subplot(212)
# plt.imshow(canny, cmap='gray')
# canny_img.set_title(f"canny edge detcetion: T1={T1}, T2={T2}")
# plt.show()


# --------------------- bilateral denoising: --------------------------

pix_hood = 5
sigma_color = 50
sigma_space = 50

bilaterl = cv.bilateralFilter(img_to_denoise, pix_hood, sigma_color, sigma_space)
plt.figure(figsize=(20, 16))
origin = plt.subplot(211)
plt.imshow(img_to_denoise, cmap='gray')
origin.set_title("Original")

denoise_img = plt.subplot(212)
plt.imshow(bilaterl, cmap='gray')
denoise_img.set_title(f"bilaterl denoise - pixel neighbourhood={pix_hood}, sigma color={sigma_color},"
                      f" sigma space={sigma_space}")
plt.show()

