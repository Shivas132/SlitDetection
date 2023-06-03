"""Module for statistic analysis of an experiment.
The assumption is that all the objects in the video are slits - the whole cleaning process has been done before.
"""
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from image_process_utils import *
from skimage import measure
from PIL import Image
import pandas as pd
import tempfile
import os

PIXEL_SIZE = 32 * 10 ** (-3)


class StatisticsModule:
    def __init__(self, video, flat_area, flat_height, flat_width, name, pxl_size=PIXEL_SIZE, resolution=1):
        self.video = normalize_to_float(video)
        self.flat_area = flat_area
        self.flat_height = flat_height
        self.flat_width = flat_width
        self.name = name
        self.pxl_size = pxl_size
        self.resolution = resolution
        self.coverage_rate_vector, self.coverage_rate_plot = self.slits_coverage_rate()
        self.avg_width, self.avg_with_plot = self.slits_width_over_time()
        self.lengths, self.length_plot = self.slits_length_over_time()
        self.temp_dir = tempfile.mkdtemp()
        self.plot_3d_0 = self.plot_3d_slit( 180, -30, 180, 0)
        self.plot_3d_1 = self.plot_3d_slit( 180, 0, 180, 1)
        self.plot_3d_2 = self.plot_3d_slit(135, -30, 180, 2)

    def plot_3d_slit(self, azim, elevation, roll, num):
        """
            Plots a 3D representation of a slit in the video.

            Args:
                num:
                video (ndarray): The video array.
                azim (float): The azimuthal angle of the view.
                elevation (float): The elevation angle of the view.
                roll (float): The roll angle of the view.

            Returns:
                None
        """
        z, y, x = np.where(self.video == 1)
        colors = z
        fig = Figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z, x, y, marker='.', s=1, c=colors, cmap='viridis')
        ax.view_init(elev=elevation, azim=azim, roll=roll)
        ax.set_xticklabels([])
        ax.set_xlabel('Time Axis')
        ax.set_ylabel('Width Axis')
        ax.set_zlabel('Height Axis')
        ax.set_title(f"azim = {azim}, elev = {elevation}, roll = {roll}")
        plot_3d_path = os.path.join(self.temp_dir, f'plot_3d_{num}.png')
        fig.savefig(plot_3d_path, format='png')
        plt.close(fig)
        image = Image.open(plot_3d_path)
        fig_0 = Figure()
        fig_0.figimage(image)
        image.close()
        os.remove(plot_3d_path)
        return fig_0

    def plot_stats_for_time(self, data, title, y_label, x_label="Time"):
        time_axis = np.arange(0, data.shape[0], 1)
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(time_axis, data)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid()
        return fig

    def slits_coverage_rate(self):
        pixel_size = self.pxl_size / self.resolution
        coverage_rate_vector = np.sum(np.sum(self.video, axis=1), axis=1) * pixel_size
        coverage_rate_vector = (coverage_rate_vector / self.flat_area) * 100
        return coverage_rate_vector,\
            self.plot_stats_for_time( coverage_rate_vector, "Slit Area Percentage Over Time", "Percentage" )

    def slits_width_over_time(self):
        pixel_size = self.pxl_size / self.resolution
        avg_width = np.mean(np.sum(self.video, axis=1), axis=1) * pixel_size
        avg_width = (avg_width / self.flat_height) * 100
        return avg_width, self.plot_stats_for_time(avg_width, "Slit Width Percentage Over Time", "Percentage")

    @staticmethod
    def sum_object_widths(frame):
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

    def slits_length_over_time(self):
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
        pixel_size = PIXEL_SIZE / self.resolution
        for frame in self.video:
            width = self.sum_object_widths(frame)
            lengths.append(width)

        lengths = np.array(lengths) * pixel_size
        lengths = (lengths / self.flat_width) * 100

        return lengths, self.plot_stats_for_time(lengths, "Slit Length Percentage Over Time", "Percentage")

    def get_plots(self):
        return [self.coverage_rate_plot,self.length_plot,self.avg_with_plot,self.plot_3d_0,self.plot_3d_1,self.plot_3d_2]
    def print_to_files(self, name):
        # data to excel
        data = {
            'Time': np.arange(1, self.video.shape[0] + 1),
            'Coverage Rate': self.coverage_rate_vector,
            'Slit Width': self.avg_width,
            'Slit Length': self.lengths,
        }
        df = pd.DataFrame(data)
        df.set_index('Time', inplace=True)
        # Save DataFrame to Excel
        path = f"{OUTPUTS}{name}"
        df.to_excel(f'{path}_slits_data.xlsx')

        # Save plots to PDF
        with PdfPages(f'{path}_slits_plots.pdf') as pdf:
            pdf.savefig(self.coverage_rate_plot)
            pdf.savefig(self.length_plot)
            pdf.savefig(self.avg_with_plot)
            pdf.savefig(self.plot_3d_0)
            pdf.savefig(self.plot_3d_1)
            pdf.savefig(self.plot_3d_2)

        # Remove temporary files and directory
        os.rmdir(self.temp_dir)
