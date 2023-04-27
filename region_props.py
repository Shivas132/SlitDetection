"""Module to find slits properties."""

import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure, data
import numpy as np
import pandas as pd
from skimage.morphology import closing, square
import cv2 as cv
from image_process_utils import *
from denoising import denoise_video
from difference_finding import deltas_video

# video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\outputs\exp_0_deltas_or_thresh0.02.dat"
# video = frames_as_matrix_from_binary_file(video_path, offset=False)

# --- create interactive image of region props for last frame: ---

# frame_num = 127
# frame = video[frame_num]
#
# morph_size = 3
# closed = closing(frame, square(morph_size))


def region_props_hover(img: np.array, props: list[str], title: str = "") -> None:
    """Show region properties in an interactive image.

    Open an HTML page with the image where the regions are marked.
    Properties of region are displayed by hovering the region with the mouse.

    Args:
        img: image to find regions in.
        props: list of properties to display.
        title: title to the image. by default, it's an empty string.

    """
    labels = measure.label(img)
    fig = px.imshow(img, binary_string=True, title=title)
    fig.update_traces(hoverinfo='skip')
    properties = measure.regionprops(labels, img)
    for index in range(1, labels.max()):
        label_i = properties[index].label
        contour = measure.find_contours(labels == label_i)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop in props:
            hoverinfo += f'<b>{prop}: {getattr(properties[index], prop):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)


properties = ['area', 'eccentricity', 'perimeter', 'axis_major_length', 'orientation']

# box = np.zeros((30, 30), dtype='int')
# box[2:12, 2:4] = 255
# box[15:17, 4:20] = 255
# regs = measure.regionprops(box)
# region_props_hover(box, ['area', 'orientation'])


def imfill(img: np.array) -> np.array:
    """Fill holes in objects in an image using flood fill algorithm.

    Args:
        img: must be binary image, and type of values is int.

    Returns:
        np.array: the image with the holes filled.
    """
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    im_ff = img.copy()
    cv.floodFill(im_ff, mask, (0, 0), MAX_GRAY_VAL)
    im_ff = cv.bitwise_not(im_ff)
    return img | im_ff


# --- create imfill video: ---
# video = normalize_to_int(video)
# imfills = np.empty(video.shape)
# for i in range(video.shape[0]):
#     imfills[i] = imfill(video[i])

# imfills = normalize_to_float(imfills)
# save_video(imfills, "exp_0_imfill_float")


# --- remove regions that are too small to be a slit: ---
def deltas_with_noise_removal(data: np.array, thresh: float = 0.02) -> np.array:
    """Create deltas video from data, with removal of non-slit objects.

    The method gets a video and create a for every frame a frame that contains the differences from the previous one.
    In each new frame, a white pixel is a pixel that has changed, and a black one is a pixel that hasn't.

    Additionally, objects that had been identified as no slits are being blacken.
    Identification is Based on a minimal area of the object, and range of angles at which a slit may appear.

    Args:
        data: input video.
        thresh: threshold for the thresholding process for each frame.

    Returns:
        np.array: a new video with only the filtered differences.
    """
    frames_num, rows, cols = data.shape
    deltas = np.empty((frames_num, rows, cols), dtype=np.uint8)
    sum_of_deltas = np.zeros((rows, cols), np.float64)
    for i in range(1, frames_num):
        delta_img = data[i] - data[i-1]
        delta_img = thresholding(delta_img, thresh)
        sum_of_deltas = cv.bitwise_or(sum_of_deltas, delta_img)
        deltas[i] = sum_of_deltas
        # remove noise by area, orientation, eccentricity:
        deltas[i] = imfill(deltas[i])
        labels = measure.label(deltas[i])
        regs = measure.regionprops(labels, deltas[i])
        # properties to recognize a slit:
        min_area = 4
        orientation = 0.01
        min_eccentricity = 0.7
        for reg in regs:
            if reg.area < min_area or np.abs(reg.orientation) < orientation or reg.eccentricity < min_eccentricity:
                min_r, min_c, max_r, max_c = reg.bbox
                deltas[i][min_r:max_r, min_c:max_c] = 0
    return deltas


def noise_remove_by_props(data, min_area=4, orientation=0.01, min_eccentricity=0.7):
    frames_num, rows, cols = data.shape
    for i in range(frames_num):
        # remove noise by area, orientation, eccentricity:
        data[i] = imfill(data[i])
        labels = measure.label(data[i])
        regs = measure.regionprops(labels, data[i])
        # properties to recognize a slit:
        for reg in regs:
            if reg.area < min_area or np.abs(reg.orientation) < orientation or reg.eccentricity < min_eccentricity:
                min_r, min_c, max_r, max_c = reg.bbox
                data[i][min_r:max_r, min_c:max_c] = 0
    return data


# video = normalize_to_int(video)
# deltas = deltas_with_noise_removal(video)
# save_video(deltas, "exp_0_denoised_orientation_0.01_size_4_eccen_0.7_filter")
# props_remove = noise_remove_by_props(video)
# save_video(props_remove, "exp_0_deltas_imfilled_orientation_0.01_size_4_eccen_0.7_filter")
# region_props_hover(deltas[117], ['area', 'orientation', 'eccentricity'])


# frames = video[-2:]
# frames = normalize_to_int(frames, MAX_GRAY_VAL)
# comp_props = ['centroid', 'bbox', 'area', 'coords']
# max_centers_dist = 2

# filled = np.array([imfill(frame) for frame in frames])
# labeled_frames = [measure.label(frame) for frame in filled]
# tables = [measure.regionprops_table(frame, properties=comp_props) for frame in labeled_frames]
# tables = [pd.DataFrame(table) for table in tables]
# regions = [measure.regionprops(frame) for frame in frames]


# -------------- full process for exp_1: --------------
exp_1_path = f"{EXP_1}exp.dat"
data = frames_as_matrix_from_binary_file(exp_1_path)
data = normalize_to_int(data)
data = denoise_video(data, h=3, template_window_size=7, search_window_size=21)
data = normalize_to_float(data)

# experiment of thresholds:
# thresh = [0.01, 0.02, 0.03, 0.04]
# for thresh in thresh:
#     data = deltas_video(data, thresh=thresh)
#     data = noise_remove_by_props(data)
#     save_video(data, f'exp_1_deltas_thresh_{thresh}_orientation_0.01_size_4_eccen_0.7_filter', DELTAS)

# ideal: 0.05

# experiment of min_area:
# areas = [0.5, 1, 2, 3.5]
# for area in areas:
#     data = deltas_video(data, 0.04)
#     data = noise_remove_by_props(data, min_area=area)
#     save_video(data, f'exp_1_deltas_thresh_0.04_orientation_0.01_size_{area}_eccen_0.7_filter', DELTAS)

# didn't give good results


# -------------- exp_0 full pre-process: --------------
# exp_0_path = f"{EXP_0}exp.dat"
# data = frames_as_matrix_from_binary_file(exp_0_path)
# data = normalize_to_int(data)
# data = denoise_video(data, h=3, template_window_size=7, search_window_size=21)
# data = normalize_to_float(data)
# data = deltas_video(data, thresh=0.02)
# data = noise_remove_by_props(data)
# save_video(data, 'exp_0_preprocess_seperated_orientation_0.01_size_4_eccen_0.7_filter', DELTAS)

