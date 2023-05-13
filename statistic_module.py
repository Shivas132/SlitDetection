"""Module for statistic analysis of an experiment.
The assumptions is that all the objects in the video are slits - the whole cleaning process has been done before.
"""
import numpy as np
import matplotlib.pyplot as plt
from image_process_utils import *
from skimage import measure
import cv2 as cv


def plot_stats_for_time(data, title, y_label, x_label="Time"):
    time_axis = np.arange(0, data.shape[0], 1)
    plt.plot(time_axis, data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()


def slits_coverage_rate(video, plot_graph=False):
    frames, rows, cols = video.shape
    total_pxl_num = rows * cols
    coverage_rate_vector = np.sum(np.sum(video, axis=1), axis=1)
    coverage_rate_vector = (coverage_rate_vector / total_pxl_num) * 100
    if plot_graph:
        plot_stats_for_time(coverage_rate_vector, "objects per frame percentage", "Percentage")

    return coverage_rate_vector


def slits_number(video, plot_graph=False):
    video = normalize_to_int(video)
    frames, rows, cols = video.shape
    objects_num_vector = np.zeros(frames, dtype=int)
    for i in range(frames):
        contours, _ = cv.findContours(video[i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        objects_num_vector[i] = len(contours)
    if plot_graph:
        plot_stats_for_time(objects_num_vector, "Number of objects in a frame", "# Objects")
    return objects_num_vector


def slits_area_over_time(video):
    video = normalize_to_int(video)
    frames, rows, cols = video.shape
    labels = measure.label(video[frames-1])
    slit_areas = np.zeros((len(labels)))



video_path = f"{DELTAS}exp_1_deltas_thresh_0.05_orientation_0.01_size_4_eccen_0.7_filter.dat"
data = frames_as_matrix_from_binary_file(video_path, offset=False)
# slits_coverage_rate(data, True)
# obj_n = slits_number(data, True)
