"""Module for statistic analysis of an experiment.
The assumption is that all the objects in the video are slits - the whole cleaning process has been done before.
"""
from image_process_utils import *
from matplotlib.figure import Figure

from skimage import measure
import pandas as pd
PIXEL_SIZE = 32 * 10 ** (-3)


def plot_stats_for_time(data, title, y_label, x_label="Time"):
    """
        Plots statistics over time.

        Args:
            data (array-like): The data to plot.
            title (str): The title of the plot.
            y_label (str): The label for the y-axis.
            x_label (str, optional): The label for the x-axis. Defaults to "Time".

        Returns:
            fig (matplotlib.figure.Figure): The figure containing the plot.
    """
    # Plotting code (unchanged)
    time_axis = np.arange(0, data.shape[0], 1)
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot(time_axis, data)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    return fig


def slits_coverage_rate(video, flat_area, resolution=1, pxl_size=PIXEL_SIZE, plot_graph=False):
    """
        Calculates the coverage rate of slits in a video.

        Args:
            video (ndarray): 3D array representing the video frames.
            flat_area (float): Area of the flat region.
            resolution (int, optional): Resolution of the video frames. Defaults to 1.
            pxl_size (float, optional): Pixel size. Defaults to PIXEL_SIZE.
            plot_graph (bool, optional): Whether to plot the coverage rate graph. Defaults to False.

        Returns:
            ndarray: Array representing the coverage rate of slits over time.
        """
    pixel_size = pxl_size / resolution
    coverage_rate_vector = np.sum(np.sum(video, axis=1), axis=1) * pixel_size
    coverage_rate_vector = (coverage_rate_vector / flat_area) * 100
    if plot_graph:
        return plot_stats_for_time(coverage_rate_vector, "Slit Area Percentage Over Time", "Percentage")

    return coverage_rate_vector


def slits_width_over_time(video, flat_height, pxl_size=PIXEL_SIZE, resolution=1, plot_graph=False):
    """
        Calculates the average width of slits over time.

        Args:
            video (ndarray): 3D array representing the video frames.
            flat_height (float): Height of the flat region.
            pxl_size (float, optional): Pixel size. Defaults to PIXEL_SIZE.
            resolution (int, optional): Resolution of the video frames. Defaults to 1.
            plot_graph (bool, optional): Whether to plot the width graph. Defaults to False.

        Returns:
            ndarray: Array representing the average width of slits over time.
        """
    pixel_size = pxl_size / resolution
    avg_width = np.mean(np.sum(video, axis=1), axis=1) * pixel_size
    avg_width = (avg_width / flat_height) * 100
    if plot_graph:
        return plot_stats_for_time(avg_width, "Slit Width Percentage Over Time", "Percentage")

    return avg_width


def sum_object_widths(frame):
    """
        Computes the sum of widths of non-overlapping objects in a frame.

        Args:
            frame (ndarray): 2D binary array representing the frame.

        Returns:
            int: The sum of widths of non-overlapping objects.
        """
    labels = measure.label(frame)
    regs = measure.regionprops(labels)
    regs = sorted(regs, key=lambda x: x.area, reverse=True)
    total_width = 0

    for i, region in enumerate(regs):
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col + 1

        is_overlapping = False
        for j in range(i):
            prev_region = regs[j]
            prev_min_row, prev_min_col, prev_max_row, prev_max_col = prev_region.bbox

            if min_col <= prev_max_col and max_col >= prev_min_col:
                is_overlapping = True
                break

        if not is_overlapping:
            total_width += width

    return total_width


def slits_length_over_time(video, flat_width, pxl_size=PIXEL_SIZE, resolution=1, plot_graph=False):
    """
        Calculates the length of slits over time.

        Args:
            video (ndarray): 3D array representing the video frames.
            flat_width (float): Width of the flat region.
            pxl_size (float, optional): Pixel size. Defaults to PIXEL_SIZE.
            resolution (int, optional): Resolution of the video frames. Defaults to 1.
            plot_graph (bool, optional): Whether to plot the length graph. Defaults to False.

        Returns:
            ndarray: Array representing the length of slits over time.
        """
    lengths = []
    pixel_size = pxl_size / resolution
    for frame in video:
        width = sum_object_widths(frame)
        lengths.append(width)

    lengths = np.array(lengths) * pixel_size
    lengths = (lengths / flat_width) * 100

    if plot_graph:
        return plot_stats_for_time(lengths, "Slit Length Percentage Over Time", "Percentage")

    return lengths


def find_4_extreme_points(video):
    points_list = []

    for frame in video:
        object_indices = np.where(frame == 1)

        if object_indices[0].size == 0:
            points_list.append([(0, 0)] * 4)
        else:
            leftmost_index = np.argmin(object_indices[1])
            rightmost_index = np.argmax(object_indices[1])
            topmost_index = np.argmin(object_indices[0])
            bottommost_index = np.argmax(object_indices[0])

            leftmost_point = (object_indices[0][leftmost_index], object_indices[1][leftmost_index])
            rightmost_point = (object_indices[0][rightmost_index], object_indices[1][rightmost_index])
            topmost_point = (object_indices[0][topmost_index], object_indices[1][topmost_index])
            bottommost_point = (object_indices[0][bottommost_index], object_indices[1][bottommost_index])

            points_list.append([leftmost_point, rightmost_point, topmost_point, bottommost_point])

    return np.array(points_list)


video = frames_as_matrix_from_binary_file(f"{OUTPUTS}exp_deltas_thresh=0.06_final_results.dat", offset=False)
# # # find_4_extreme_points()
# slits_coverage_rate(video, resolution=2, flat_area=1570.8, plot_graph=True)
# slits_width_over_time(video, resolution=1, flat_height=8, plot_graph=True)
# slits_length_over_time(video, resolution=1, flat_width=8, plot_graph=True)


def get_stats(video,resolution,flat_area,flat_height,flat_width):
    return [slits_coverage_rate(video, resolution=resolution, flat_area=flat_area, plot_graph=True),
            slits_width_over_time(video, resolution=resolution, flat_height=flat_height, plot_graph=True),
            slits_length_over_time(video, resolution=resolution, flat_width=flat_width, plot_graph=True)]

def collect_data_and_print_to_files(video, flat_area, flat_height, flat_width, pxl_size=PIXEL_SIZE, resolution=1):
    from matplotlib.backends.backend_pdf import PdfPages

    coverage_rate = slits_coverage_rate(video, flat_area, resolution, pxl_size, plot_graph=False)
    width_over_time = slits_width_over_time(video, flat_height, pxl_size, resolution, plot_graph=False)
    length_over_time = slits_length_over_time(video, flat_width, pxl_size, resolution, plot_graph=False)

    data = {
        'Time': np.arange(1, video.shape[0] + 1),
        'Coverage Rate': coverage_rate,
        'Slit Width': width_over_time,
        'Slit Length': length_over_time,
    }
    df = pd.DataFrame(data)
    df.set_index('Time', inplace=True)
    # Save DataFrame to Excel
    df.to_excel('slits_data.xlsx')

    coverage_rate_plot = slits_coverage_rate(video, flat_area, resolution, pxl_size, plot_graph=True)
    width_over_time_plot = slits_width_over_time(video, flat_height, pxl_size, resolution, plot_graph=True)
    length_over_time_plot = slits_length_over_time(video, flat_width, pxl_size, resolution, plot_graph=True)

    data = {
        'Time': np.arange(1, video.shape[0] + 1),
        'Coverage Rate': coverage_rate_plot,
        'Slit Width': width_over_time_plot,
        'Slit Length': length_over_time_plot,
    }

    df = pd.DataFrame(data)
    df.set_index('Time', inplace=True)

    # Save DataFrame to Excel
    df.to_excel('slits_data.xlsx', sheet_name='Data')

    # Save plots to PDF
    with PdfPages('slits_plots.pdf') as pdf:
        pdf.savefig(coverage_rate_plot)
        pdf.savefig(width_over_time_plot)
        pdf.savefig(length_over_time_plot)

    plt.close('all')  # Close all open plots
    print("Data has been saved to 'slits_data.xlsx'.")
    print("Plots have been saved to 'slits_plots.pdf'.")


