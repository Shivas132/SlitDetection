from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_process_utils import frames_as_matrix_from_binary_file, normalize_to_int, save_video
from denoising import denoise_video
from region_props import create_deltas_videos, noise_remove_by_props, clean_area
from RectangleDrawer import RectangleDrawer

class App:
    def __init__(self):
        self.root = tk.Tk()
        # self.line0 = tk.Frame(self.root)
        self.line4 = tk.Frame(self.root)
        # self.line0.grid_rowconfigure(0, weight=1)
        # self.line0.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=5)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        # self.line0.grid(row=0, column=0, pady=0)
        self.line4.grid(row = 4, column = 0, pady = 0)

        # buttons
        self.select_video_button = tk.Button(self.root, text="Select Raw Video", command=self.select_raw_video)
        self.select_slit_area_button = tk.Button(self.root, text="click here when done", command=self.select_slit_area_action)
        self.create_deltas_button = tk.Button(self.root, text="create deltas videos", command=self.create_deltas_action)
        self.save_deltas_button = tk.Button(self.root, text="Save this video", command=self.save_deltas_video)
        self.save_final_button = tk.Button(self.root, text="Save", command=self.save_final_video)
        self.first_clean_button = tk.Button(self.root, text="clean Video", command=self.first_clean_helper_action)
        self.select_frame_button = tk.Button(self.root, text='Select Frame', command=self.select_first_frame_action)
        self.select_last_frame_button = tk.Button(self.root, text='Select Frame', command=self.select_last_frame_action)
        self.frame_scale = None

        self.select_video_button.grid(row=0, column=0, pady=5, padx=5)
        self.select_thresh_button = tk.Button(self.root, text="choose this threshold", command=self.select_threshold_action)
        self.second_clean_button = tk.Button(self.root, text="clean Video", command=self.second_clean_action)
        self.clean_area_button = tk.Button(self.root, text="clean this area", command=self.clean_area_action)
        self.done_cleaning_button = tk.Button(self.root, text="Done", command=self.done_cleaning_action)
        self.go_statistics_button = tk.Button(self.root, text="statistics", command=self.go_statistics_action)



        self.message_label = tk.Label(self.root, text="Please select a video file", font=("Arial", 14))
        self.message_label.grid(row=1, column=0, pady=5)
        self.fig = None
        self.ax = None
        self.ax1 = None
        self.data = None
        self.new_data = None
        self.slit_frame = None
        self.slit_frame_idx = None
        self.deltas_videos = []
        self.current_frame = 0
        self.rec = None
        self.slit_area = None
        self.slit_frame_idx = None
        self.canvas = None
        self.video_scale = None
        self.exp_name = None
        self.black_rectangle = None
        self.canvas_widget= None
        self.create_canvas()
        tk.mainloop()

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
        self.frame_scale.grid_forget()
        self.fig, (self.ax, self.ax1) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        self.ax.set_axis_off()
        self.ax1.set_axis_off()
        self.ax.imshow(self.data[0], cmap='gray')
        self.ax1.imshow(self.new_data[0], cmap='gray')
        self.frame_scale = tk.Scale(self.root, from_=0, to=127, orient=tk.HORIZONTAL, length=800)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, padx=0, pady=0, sticky="NSWE")
        self.frame_scale.configure(command=self.update_image)

    def save_name(self, path):
        filename = path.split('/')[-1]  # Get the filename from the path
        self.exp_name = '.'.join(filename.split('.')[:-1])  # Strip the extension
        print(self.exp_name)

    def update_image(self, value):
        self.current_frame = int(value)
        self.ax.images[0].set_data(self.data[self.current_frame])
        if self.new_data is not None:
            self.ax1.images[0].set_data(self.new_data[self.current_frame])
        self.fig.canvas.draw()

    def save_deltas_video(self):
        name = f"{self.exp_name}_deltas_thresh={self.cur_thresh}"
        path = save_video(self.new_data, name)
        self.print_message(f"your video saved under\n {path}")

    def save_final_video(self):
        name = f"{self.exp_name}_deltas_thresh={self.cur_thresh}_final_results"
        path = save_video(self.new_data, name)
        self.print_message(f"your video saved under\n{path}")

    def print_message(self, message):
        self.message_label.config(text=message, fg='black')
        self.root.update()

    def print_error(self, message):
        self.message_label.config(text=message, fg='red')
        self.root.update()

    def switch_deltas_video(self, value):
        value = int(value)
        self.thresh_video_index = value
        self.new_data = self.deltas_videos[value]
        treshes = [i / 1000 for i in range(0, 100, 5)]
        self.cur_thresh = treshes[value]
        self.message_label.config(text=f"tresh = {treshes[value]}")
        self.update_image(self.current_frame)
    # -----App flow------
    def select_raw_video(self):
        self.select_video_button.grid_remove()
        file_path = filedialog.askopenfilename(initialdir="/", title="Select Video",
                                               filetypes=(("dat files", "*.dat"),))
        self.save_name(file_path)
        self.data = frames_as_matrix_from_binary_file(file_path)
        self.data = normalize_to_int(self.data)
        self.ax.imshow(self.data[0], cmap='gray')
        self.fig.canvas.draw()
        self.frame_scale.grid(row=3, column=0, pady=2, sticky="N")
        self.select_first_frame()

    def select_first_frame(self):
        self.frame_scale.set(0)
        self.print_message("Please select a frame where the slit shows do to not above 110")
        self.select_frame_button.grid(row=4, column=0, pady=10)

    def select_first_frame_action(self):
        self.print_message("")
        self.slit_frame_idx = int(self.frame_scale.get())
        print(self.slit_frame_idx)
        self.slit_frame = self.data[int(self.frame_scale.get())]
        if self.slit_frame_idx >= 110:
            self.print_error("video have to be more the 10 frame")
            return
        if self.slit_frame_idx >= 5:
            self.data = self.data[self.slit_frame_idx - 5:]
            print((self.data).shape)
            self.update_image(0)
            self.frame_scale.set(0)
            self.frame_scale.configure(from_=0, to=127 - self.slit_frame_idx + 5)
        self.select_frame_button.grid_remove()
        self.root.update()
        # self.select_frame_button.grid_remove()
        self.select_last_frame()

    def select_last_frame(self):
        self.frame_scale.set(0)
        self.print_message("if wou want to cut the end of the video please select the last frame\n"
                           "(Optional, leave on zero for not choosing")
        self.select_last_frame_button.grid(row=4, column=0, pady=10)

    def select_last_frame_action(self):
        last_frame_idx = int(self.frame_scale.get())
        if last_frame_idx != 0:
            if (last_frame_idx) <= 10:
                self.print_error("video have to be more the 10 frame")
                return
            self.data = self.data[:last_frame_idx+1]
            self.update_image(0)
            self.frame_scale.set(0)
            self.frame_scale.configure(from_=0, to=last_frame_idx)
        self.root.update()
        self.select_last_frame_button.grid_remove()
        self.select_slit_area()

    def select_slit_area(self):
        self.print_message("Please select slit area")
        self.rec = RectangleDrawer(self.fig, self.ax)
        self.select_slit_area_button.grid(row=5, column=0, pady=5)

    def select_slit_area_action(self):
        self.slit_area = self.rec.ret_vals()
        self.rec.kill()
        self.rec = None
        self.select_slit_area_button.grid_remove()
        self.first_clean1()

    def first_clean1(self):
        self.message_label.config(text="Click clean to clean your video")
        self.first_clean_button.grid(row=4, column=0, pady=10)

    def first_clean_helper_action(self):
        self.first_clean_button.grid_remove()
        self.message_label.config(text="Processing video, please wait...")
        self.root.update()
        self.new_data = denoise_video(self.data)
        self.message_label.config(text="here is you video after first clean")
        self.create_new_canvas()
        self.root.update()
        self.create_deltas()

    def create_deltas(self):
        self.create_deltas_button.grid(row=4, column=0, pady=10)
        self.frame_scale.set(0)
        self.update_image(0)

    def create_deltas_action(self):
        self.create_deltas_button.grid_remove()
        self.print_message("Processing video, please wait...")
        self.create_deltas_button.grid_remove()
        self.deltas_videos = create_deltas_videos(self.new_data, self.slit_area)
        self.video_scale = tk.Scale(self.root, from_=0, to=19, orient=tk.VERTICAL, length=200, showvalue=False,
                                    command=self.switch_deltas_video)
        self.video_scale.grid(row=2, column=0, sticky='e')
        self.switch_deltas_video(0)
        self.frame_scale.set(0)
        self.update_image(0)
        self.save_deltas_button.grid(row=0, column=0, pady=2)
        self.select_threshold()

    def select_threshold(self):
        self.print_message("Use the right slider to choose your favorite threshold")
        self.select_thresh_button.grid(row=4, column=0, pady=10)

    def select_threshold_action(self):
        self.select_thresh_button.grid_remove()
        self.video_scale.grid_remove()
        self.second_clean()

    def second_clean(self):
        self.print_message("")
        self.second_clean_button.grid(row=4, column=0, pady=2)

    def second_clean_action(self):
        self.frame_scale.set(0)
        self.update_image(0)
        self.save_deltas_button.grid_remove()
        self.second_clean_button.grid_remove()
        self.print_message("Processing video please wait......")
        self.new_data = noise_remove_by_props(self.new_data)
        self.root.update()
        self.save_final_button.grid(row=0, column=0, pady=2)
        self.clean_area()

    def clean_area(self):
        self.frame_scale.set(0)
        self.update_image(0)
        self.print_message("Please select noise and click clean")
        self.clean_area_button.grid(row=4, column=0, pady=2)
        self.done_cleaning_button.grid(row=5, column=0, pady=2)
        self.black_rectangle = RectangleDrawer(self.fig, self.ax1)


    def clean_area_action(self):
        self.new_data = clean_area(self.new_data, self.black_rectangle.ret_vals())
        self.update_image(self.current_frame)
        self.root.update()

    def done_cleaning_action(self):
        self.print_message("Here is your final Video")
        self.clean_area_button.grid_remove()
        self.done_cleaning_button.grid_remove()
        self.go_statistics_button.grid(row=4, column=0, pady=2)

    def go_statistics_action(self):
        self.go_statistics_button.grid_forget()
        self.frame_scale.grid_remove()
        self.canvas_widget.grid_remove()
        self.print_message("Please enter the experiment parameters:")
        self.input_box = tk.Frame(self.root)
        self.input_box.configure(borderwidth=2, relief="solid")

        height_title = tk.Label(self.input_box, text="Model height:")
        self.height_entry = tk.Entry(self.input_box)
        width_title = tk.Label(self.input_box, text="Model width:")
        self.width_entry = tk.Entry(self.input_box)
        area_title = tk.Label(self.input_box, text="Model area:")
        self.area_entry = tk.Entry(self.input_box)
        resolution_title = tk.Label(self.input_box, text="Resolution?")
        self.resolution_entry = tk.Entry(self.input_box)

        self.input_box.grid_rowconfigure(0, weight=1)
        self.input_box.grid_rowconfigure(1, weight=1)
        self.input_box.grid_columnconfigure(0, weight=1)
        self.input_box.grid_columnconfigure(1, weight=1)
        self.input_box.grid_columnconfigure(2, weight=1)
        self.input_box.grid_columnconfigure(3, weight=1)
        height_title.grid(row=0, column=0, padx=10, pady=2)
        self.height_entry.grid(row=1, column=0, padx=10, pady=2)
        width_title.grid(row=0, column=1, padx=10, pady=2)
        self.width_entry.grid(row=1, column=1, padx=10, pady=2)
        area_title.grid(row=0, column=2, padx=10, pady=2)
        self.area_entry.grid(row=1, column=2, padx=10, pady=2)
        resolution_title.grid(row=0, column=3, padx=10, pady=2)
        self.resolution_entry.grid(row=1, column=3, padx=10, pady=2)
        self.input_box.grid(row=2, column=0, pady=2)
        self.set_size_button = tk.Button(self.root, text="done", command=self.check_size_input)
        self.set_size_button.grid(row=5, column=0, pady=20)
    def check_size_input(self):
        self.width = self.width_entry.get()
        self.height = self.height_entry.get()
        self.area = self.area_entry.get()
        self.zoom = self.resolution_entry.get()
        try:
            self.width = float(self.width)
            self.height = float(self.height)
            self.area = float(self.area)
            self.zoom = float(self.zoom)

        except:
            self.print_error("width and height have to be a number")
        finally:
            self.set_size_button.grid_forget()
            self.input_box.grid_forget()
            self.set_size_button.grid_forget()
            self.create_statistics()
    def create_statistics(self):
        self.print_message("statistics")


app = App()
