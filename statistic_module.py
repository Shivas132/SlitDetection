"""Module for statistic analysis of an experiment.
The assumptions is that all the objects in the video are slits - the whole cleaning process has been done before.
"""
import numpy as np
import matplotlib.pyplot as plt
from image_process_utils import *
from skimage import measure
import cv2 as cv


def plot_stats_for_time(data, title, y_label, x_label="Time"):
    """
        Plots statistics over time.

        Args:
            data (array-like): The data to plot.
            title (str): The title of the plot.
            y_label (str): The label for the y-axis.
            x_label (str, optional): The label for the x-axis. Defaults to "Time".

        Returns:
            None
    """
    time_axis = np.arange(0, data.shape[0], 1)
    plt.plot(time_axis, data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()


def slits_coverage_rate(video, plot_graph=False):
    """
        Calculates the coverage rate of slits in a video.

        Args:
            video (array-like): The video data.
            plot_graph (bool, optional): Whether to plot the coverage rate graph. Defaults to False.

        Returns:
            numpy.ndarray: The coverage rate vector.
    """
    frames, rows, cols = video.shape
    total_pxl_num = rows * cols
    coverage_rate_vector = np.sum(np.sum(video, axis=1), axis=1)
    coverage_rate_vector = (coverage_rate_vector / total_pxl_num) * 100
    if plot_graph:
        plot_stats_for_time(coverage_rate_vector, "objects per frame percentage", "Percentage")

    return coverage_rate_vector


def slits_number(video, plot_graph=False):
    """
       Calculates the number of slits in each frame of a video.

       Args:
           video (array-like): The video data.
           plot_graph (bool, optional): Whether to plot the number of objects graph. Defaults to False.

       Returns:
           numpy.ndarray: The objects number vector.
    """
    video = normalize_to_int(video)
    frames, rows, cols = video.shape
    objects_num_vector = np.zeros(frames, dtype=int)
    for i in range(frames):
        contours, _ = cv.findContours(video[i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        objects_num_vector[i] = len(contours)
    if plot_graph:
        plot_stats_for_time(objects_num_vector, "Number of objects in a frame", "# Objects")
    return objects_num_vector


def slits_area_over_time(video, plot_graph=False):
    """
        Calculates the area of slits over time in a video.

        Args:
            video (array-like): The video data.
            plot_graph (bool, optional): Whether to plot the slit area graph. Defaults to False.

        Returns:
            numpy.ndarray: The slit areas over time.
    """
    video = normalize_to_int(video)
    frames, rows, cols = video.shape
    labels = measure.label(video[frames-1])
    regs = measure.regionprops(labels, video[frames-1])
    slit_areas = np.zeros((frames, 1))
    for region in regs:
        mask = np.ones_like(video, dtype=bool)
        mask[:, region.coords[:, 0], region.coords[:, 1]] = False
        masked_video = video.copy()
        masked_video[mask] = 0
        area = np.sum(np.sum(masked_video, axis=2), axis=1).reshape((frames, 1))
        slit_areas = np.concatenate((slit_areas, area), axis=1)

    slit_areas = slit_areas[:, 1:]
    if plot_graph:
        plt.plot(slit_areas)
        plt.title("Slit Areas Growth Over Time")
        plt.xlabel("Time")
        plt.ylabel("Area")
        plt.grid()
        plt.show()
    return slit_areas



# video_path = f"{DELTAS}exp_1_deltas_thresh_0.05_orientation_0.01_size_4_eccen_0.7_filter.dat"
# data = frames_as_matrix_from_binary_file(video_path, offset=False)
# # slits_coverage_rate(data, True)
# # obj_n = slits_number(data, True)

video = frames_as_matrix_from_binary_file(f"{OUTPUTS}exp_1_only_slit.dat", offset=False)
slits_area_over_time(video, True)
# slits_number(video, True)
slits_coverage_rate(video, True)

# TODO: calculate length and width. problem: The slits do not build continuously throughout the video
