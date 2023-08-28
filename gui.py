import os
from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_process_utils import frames_as_matrix_from_binary_file, normalize_to_int, save_video
from denoising import denoise_video
from region_props import create_deltas_videos, noise_remove_by_props, clean_area
from RectangleDrawer import RectangleDrawer
from statistic_module import StatisticsModule
from remove_non_slit_objects import blocks_objects_filtering
from paths import OUTPUTS
from mpl_toolkits.mplot3d import Axes3D


class App:
    """A graphical user interface application for Slit Detector.

        This class represents the main application for the Slit Detector program. It provides a graphical user interface
        (GUI) using the tkinter library, allowing users to interact with the program and perform various actions on video
        data.

    """
    def __init__(self):
        # init window and grid
        self.root = tk.Tk()
        self.root.title("Slit Detector")
        self.line0 = tk.Frame(self.root)
        self.line4 = tk.Frame(self.root)
        self.line0.grid_rowconfigure(0, weight=1)
        self.line0.grid_rowconfigure(1, weight=1)
        self.line0.grid_columnconfigure(0, weight=1)
        self.line0.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=5)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_rowconfigure(7, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        # self.line0.grid(row=0, column=0, pady=0)
        self.line4.grid(row = 4, column = 0, pady = 0)

        # buttons
        self.select_video_button = tk.Button(self.root, text="Select Raw Video", command=self.select_raw_video)
        self.select_slit_area_button = tk.Button(self.root, text="click here when done", command=self.select_slit_area_action)
        self.create_deltas_button = tk.Button(self.root, text="create deltas videos", command=self.create_deltas_action)
        self.save_deltas_button = tk.Button(self.root, text="Save this video", command=self.save_deltas_video)
        self.save_final_button = tk.Button(self.root, text="Save", command=self.save_final_video)
        self.first_clean_button = tk.Button(self.root, text="clean Video", command=self.first_clean_action)
        self.select_frame_button = tk.Button(self.root, text='Select Frame', command=self.select_first_frame_action)
        self.select_last_frame_button = tk.Button(self.root, text='Select Frame', command=self.select_last_frame_action)
        self.select_video_button.grid(row=3, column=0, pady=5, padx=5)
        self.select_thresh_button = tk.Button(self.root, text="choose this threshold", command=self.select_threshold_action)
        self.second_clean_button = tk.Button(self.root, text="clean Video", command=self.second_clean_action)
        self.clean_area_button = tk.Button(self.root, text="clean this area", command=self.clean_area_action)
        self.done_cleaning_button = tk.Button(self.root, text="Done", command=self.done_cleaning_action)
        self.go_statistics_button = tk.Button(self.root, text="statistics", command=self.go_statistics_action)
        self.save_data_to_file_button = tk.Button(self.root, text="Save data to files", command=self.save_data)

        # labels
        self.main_messege = tk.Label(self.line0, text="Welcome to Slit Detector!", font=("Arial", 18))
        self.main_messege.grid(row=1, column=0, pady=5,sticky="N")
        self.sub_main_messege = tk.Label(self.line0, text="Please select a video file", font=("Arial", 14))
        self.sub_main_messege.grid(row=2, column=0, pady=5, sticky="S")
        self.line0.grid(row=1, column=0, pady=20)

        self.frame_scale = None
        self.fig = None
        self.ax = None
        self.ax1 = None
        self.data = None
        self.new_data = None
        self.slit_frame_idx = None
        self.rec = None
        self.slit_area = None
        self.slit_frame_idx = None
        self.canvas = None
        self.threshold_scale = None
        self.exp_name = None
        self.black_rectangle = None
        self.canvas_widget = None
        self.deltas_videos = []
        self.current_figure_index = 0
        self.current_frame = 0
        self.create_canvas()
        self.figures = []

        tk.mainloop()

    # utils functions
    def create_canvas(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.frame_scale = tk.Scale(self.root, from_=0, to=127, orient=tk.HORIZONTAL, length=400)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, padx=0, pady=0, sticky="NSWE")
        self.frame_scale.configure(command=self.update_image)

    def create_new_canvas(self):
        self.canvas_widget.grid_forget()
        # self.frame_scale.grid_forget()
        self.fig, (self.ax, self.ax1) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        self.ax.set_axis_off()
        self.ax1.set_axis_off()
        self.ax.imshow(self.data[0], cmap='gray')
        self.ax1.imshow(self.new_data[0], cmap='gray')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, padx=0, pady=0, sticky="NSWE")
        self.frame_scale.configure(command=self.update_image)

    def save_name(self, path):
        filename = path.split('/')[-1]  # Get the filename from the path
        self.exp_name = '.'.join(filename.split('.')[:-1])  # Strip the extension

    def update_image(self, value):
        self.current_frame = int(value)
        self.ax.images[0].set_data(self.data[self.current_frame])
        if self.new_data is not None:
            self.ax1.images[0].set_data(self.new_data[self.current_frame])
        self.fig.canvas.draw()

    def save_deltas_video(self):
        name = f"{self.exp_name}_deltas_thresh={self.cur_thresh}"
        path = save_video(self.new_data, name)
        self.print_sub_main_message(f"your video saved under\n {path}")

    def save_final_video(self):
        name = f"{self.exp_name}_deltas_thresh={self.cur_thresh}_final_results"
        path = save_video(self.new_data, name)
        self.print_sub_main_message(f"your video saved under\n{path}")

    def print_main_message(self, message):
        self.main_messege.config(text=message, fg='black')
        self.root.update()

    def print_sub_main_message(self, message):
        self.sub_main_messege.config(text=message, fg='black')
        self.root.update()

    def print_error(self, message):
        self.sub_main_messege.config(text=message, fg='red')
        self.root.update()

    def switch_deltas_video(self, value):
        value = float(value)
        self.cur_thresh = value

        self.new_data = self.deltas_videos[int(value*20)]
        # thresholds = [i / 1000 for i in range(0, 100, 5)]
        # self.main_messege.config(text=f"threshold = {thresholds[value]}")
        self.update_image(self.current_frame)

    # -----App flow------
    def select_raw_video(self):
        project_folder = os.path.dirname(os.path.abspath(__file__))
        file_path = filedialog.askopenfilename(initialdir=project_folder, title="Select Video",
                                               filetypes=(("dat files", "*.dat"),))
        try:
            self.data = frames_as_matrix_from_binary_file(file_path)
            self.data = normalize_to_int(self.data)
            self.save_name(file_path)
            self.select_video_button.grid_remove()
        except:
            self.print_error("can't read this file please select 128/400/250 raw data")
            return
        self.ax.imshow(self.data[0], cmap='gray')
        self.fig.canvas.draw()
        self.frame_scale.grid(row=4, column=0,pady=2, sticky="N")
        self.select_first_frame()

    def select_first_frame(self):
        self.frame_scale.set(0)
        self.print_main_message("Please select the first frame where the slit appear")
        self.print_sub_main_message("don't choose frame above 110")
        self.select_frame_button.grid(row=5, column=0, pady=10)

    def select_first_frame_action(self):
        self.print_sub_main_message("")
        self.slit_frame_idx = int(self.frame_scale.get())
        if self.slit_frame_idx >= 110:
            self.print_error("video have to be more the 10 frame")
            return
        if self.slit_frame_idx >= 5:
            self.data = self.data[self.slit_frame_idx - 5:]
            self.update_image(0)
            self.frame_scale.set(0)
            self.frame_scale.configure(from_=0, to=127 - self.slit_frame_idx + 5)
        self.select_frame_button.grid_remove()
        self.root.update()
        # self.select_frame_button.grid_remove()
        self.select_last_frame()

    def select_last_frame(self):
        self.frame_scale.set(0)
        self.print_main_message("Please choose the last frame of the video")
        self.print_sub_main_message("optional leave on zero elsewhere")
        self.select_last_frame_button.grid(row=5, column=0, pady=10)

    def select_last_frame_action(self):
        last_frame_idx = int(self.frame_scale.get())
        if last_frame_idx != 0:
            if (last_frame_idx) <= 10:
                self.print_error("video has to be more the 10 frame")
                return
            self.data = self.data[:last_frame_idx+1]
            self.update_image(0)
            self.frame_scale.set(0)
            self.frame_scale.configure(from_=0, to=last_frame_idx)
        self.root.update()
        self.select_last_frame_button.grid_remove()
        self.select_slit_area()

    def select_slit_area(self):
        self.print_main_message("Please select slit area")
        self.print_sub_main_message("use your mouse to draw a rectangle around the estimated area")
        self.rec = RectangleDrawer(self.fig, self.ax)
        self.select_slit_area_button.grid(row=5, column=0, pady=5)

    def select_slit_area_action(self):
        self.slit_area = self.rec.ret_vals()
        self.rec.kill()
        self.rec = None
        self.select_slit_area_button.grid_remove()
        self.first_clean()

    def first_clean(self):
        self.print_main_message("Click clean to clean your video")
        self.print_sub_main_message("")
        self.first_clean_button.grid(row=5, column=0, pady=10)

    def first_clean_action(self):
        self.first_clean_button.grid_remove()
        self.print_main_message("Processing video, please wait...")
        self.root.update()
        self.new_data = denoise_video(self.data)
        self.print_main_message("The video after first clean")
        self.print_sub_main_message("Source on the left, cleaned on the right")
        self.create_new_canvas()
        self.root.update()
        self.create_deltas()

    def create_deltas(self):
        self.create_deltas_button.grid(row=5, column=0, pady=10)
        self.frame_scale.set(0)
        self.update_image(0)

    def create_deltas_action(self):
        self.create_deltas_button.grid_remove()
        self.print_main_message("Processing video, please wait...")
        self.print_sub_main_message("")
        self.create_deltas_button.grid_remove()
        self.deltas_videos = create_deltas_videos(self.new_data, self.slit_area)
        self.threshold_scale = tk.Scale(self.root, from_=0, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, length=400,
                                        showvalue=True,
                                        command=self.switch_deltas_video)
        self.thresh_scale_label = tk.Label(self.root, text="threshold scale", font=("Arial", 10))
        self.thresh_scale_label.grid(row=5, column=0, sticky="N",pady=0)
        self.threshold_scale.grid(row=6, column=0, sticky='S', pady=0)
        self.frame_scale_label = tk.Label(self.root, text="frames scale", font=("Arial", 10))
        self.frame_scale_label.grid(row=3, column=0, pady=0,sticky="S")
        self.switch_deltas_video(0)
        self.frame_scale.set(0)
        self.update_image(0)
        self.save_deltas_button.grid(row=0, column=0, pady=5)
        self.select_threshold()

    def select_threshold(self):
        self.print_main_message("Please select an appropriate threshold")
        self.print_sub_main_message("Use the threshold scale below and choose your favorite threshold")
        self.select_thresh_button.grid(row=7, column=0, pady=10)

    def select_threshold_action(self):
        self.frame_scale_label.grid_forget()
        self.thresh_scale_label.grid_forget()
        self.threshold_scale.grid_forget()
        self.select_thresh_button.grid_remove()
        self.threshold_scale.grid_remove()
        self.second_clean()

    def second_clean(self):
        self.print_main_message("Click clean for automate clean the binary video")
        self.print_sub_main_message("")
        self.second_clean_button.grid(row=5, column=0, pady=2)

    def second_clean_action(self):
        self.frame_scale.set(0)
        self.update_image(0)
        self.save_deltas_button.grid_remove()
        self.second_clean_button.grid_remove()
        self.print_main_message("Processing video, please wait...")
        self.print_sub_main_message("")
        self.new_data = noise_remove_by_props(self.new_data)
        self.new_data = blocks_objects_filtering(self.new_data,0)
        self.root.update()
        self.save_final_button.grid(row=0, column=0, pady=2)
        self.clean_area()

    def clean_area(self):
        self.frame_scale.set(0)
        self.update_image(0)
        self.print_main_message("Additional manual clean")
        self.print_sub_main_message("Use your mouse to draw a rectangle over noise areas and click the button below")
        self.clean_area_button.grid(row=5, column=0, pady=2)
        self.done_cleaning_button.grid(row=6, column=0, pady=20)
        self.black_rectangle = RectangleDrawer(self.fig, self.ax1)

    def clean_area_action(self):
        self.new_data = clean_area(self.new_data, self.black_rectangle.ret_vals())
        self.update_image(self.current_frame)
        self.root.update()

    def done_cleaning_action(self):
        self.print_main_message("Here is your final Video")
        self.print_sub_main_message("")
        self.black_rectangle.kill()
        self.clean_area_button.grid_remove()
        self.done_cleaning_button.grid_remove()
        self.go_statistics_button.grid(row=5, column=0, pady=2)

    def go_statistics_action(self):
        self.save_final_button.grid_forget()
        self.go_statistics_button.grid_forget()
        self.frame_scale.grid_remove()
        self.canvas_widget.grid_remove()
        self.print_main_message("Please enter the experiment parameters:")
        self.input_box = tk.Frame(self.root)
        self.input_box.configure(borderwidth=2, relief="solid")

        height_title = tk.Label(self.input_box, text="Model height:")
        self.height_entry = tk.Entry(self.input_box)
        width_title = tk.Label(self.input_box, text="Model width:")
        self.width_entry = tk.Entry(self.input_box)
        area_title = tk.Label(self.input_box, text="Model area:")
        self.area_entry = tk.Entry(self.input_box)
        magnitude_title = tk.Label(self.input_box, text="magnitude")
        self.magnitude_entry = tk.Entry(self.input_box)

        self.input_box.grid_rowconfigure(0, weight=1)
        self.input_box.grid_rowconfigure(1, weight=1)
        self.input_box.grid_columnconfigure(0, weight=1)
        self.input_box.grid_columnconfigure(1, weight=1)
        self.input_box.grid_columnconfigure(2, weight=1)
        self.input_box.grid_columnconfigure(3, weight=1)
        height_title.grid(row=0, column=0, padx=10, pady=5)
        self.height_entry.grid(row=1, column=0, padx=10, pady=5)
        width_title.grid(row=0, column=1, padx=10, pady=5)
        self.width_entry.grid(row=1, column=1, padx=10, pady=5)
        area_title.grid(row=0, column=2, padx=10, pady=5)
        self.area_entry.grid(row=1, column=2, padx=10, pady=5)
        magnitude_title.grid(row=0, column=3, padx=10, pady=5)
        self.magnitude_entry.grid(row=1, column=3, padx=10, pady=5)
        self.input_box.grid(row=2, column=0, pady=5)
        self.set_size_button = tk.Button(self.root, text="done", command=self.check_size_input)
        self.set_size_button.grid(row=5, column=0, pady=20)

    def check_size_input(self):
        self.height = self.height_entry.get()
        self.width = self.width_entry.get()
        self.area = self.area_entry.get()
        self.magnitude = self.magnitude_entry.get()
        try:
            self.height = float(self.height)
            self.width = float(self.width)
            self.area = float(self.area)
            self.magnitude = float(self.magnitude)
        except:
            self.print_error("inputs has to be numbers")
            return
        self.set_size_button.grid_forget()
        self.input_box.grid_forget()
        self.set_size_button.grid_forget()
        self.create_statistics()

    def create_statistics(self):
        self.stats_module = StatisticsModule(self.new_data, self.area, self.height, self.width, self.exp_name)
        self.print_main_message("Statistics:")
        self.print_sub_main_message("")
        self.nav_box = tk.Frame(self.root)
        # Create the buttons
        self.left_button = tk.Button(self.nav_box, text="<", command=self.show_previous_figure)
        self.right_button = tk.Button(self.nav_box, text=">", command=self.show_next_figure)

        self.left_button.grid(row=0, column=0, padx=10, pady=5)
        self.right_button.grid(row=0, column=1, padx=10, pady=5)
        self.nav_box.grid(row=4,column=0)
        self.figures= self.stats_module.get_plots()
        self.show_current_figure()
        self.save_data_to_file_button.grid(row=5,column=0)
        self.root.update()

    def save_data(self):
        name = f"{self.exp_name}_deltas_thresh={self.cur_thresh}"
        self.stats_module.print_to_files(name)
        self.print_sub_main_message(f"your video data saved under: {OUTPUTS}{name}")

    def show_previous_figure(self):
        self.current_figure_index =(self.current_figure_index-1) % len(self.figures)
        self.show_current_figure()

    def show_next_figure(self):
        self.current_figure_index = (self.current_figure_index+1) % len(self.figures)
        self.show_current_figure()

    def show_current_figure(self):
        if self.figures:
            current_figure = self.figures[self.current_figure_index]
            canvas = FigureCanvasTkAgg(current_figure, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().grid(row=3, column=0)
