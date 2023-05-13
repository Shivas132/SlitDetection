from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_process_utils import frames_as_matrix_from_binary_file, normalize_to_int, show_frame
from denoising import denoise_video, denoise_video_selected_frame, denoise2
from region_props import choose_thresh
from RectangleDrawer import RectangleDrawer


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4,weight=1)
        self.root.grid_rowconfigure(5,weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.clean_button = tk.Button(self.root, text="clean Video", command=self.first_clean)
        self.select_frame_button = tk.Button(self.root, text='Select Frame', command=self.choose_frame)
        self.frame_scale = tk.Scale(self.root, from_=0, to=127, orient=tk.HORIZONTAL, length=400)
        self.frame_scale.configure(command=self.update_image)
        self.select_video_button = tk.Button(self.root, text="Select Video", command=self.select_video)
        # self.select_slit_button = tk.Button(self.root, text="Select Slit", command=self.select_slit_area)
        self.select_slit_button_1 = tk.Button(self.root, text="click here when done", command=self.get_ret_vals)
        self.tresh_button = tk.Button(self.root, text="create treshes", command=self.show_tresh)
        self.select_video_button.grid(row = 0, column = 0, pady = 5)
        self.message_label = tk.Label(self.root, text="Please select a video file",font=("Arial", 14))
        self.message_label.grid(row = 1, column = 0, pady = 5)
        self.fig = None
        self.ax = None
        self.ax1 = None
        self.create_canvas()
        self.data = None
        self.new_data = None
        self.slit_frame = None
        self.slit_frame_idx = None
        self.tresh_videos = []
        self.current_frame = 0
        self.rec = None
        self.slit_area = None
        self.slit_frame_idx =None
        self.canvas = None
        tk.mainloop()

    def select_video(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select Video",
                                               filetypes=(("dat files", "*.dat"),))
        self.data = frames_as_matrix_from_binary_file(file_path)
        self.data = normalize_to_int(self.data)
        self.ax.imshow(self.data[0], cmap='gray')
        self.fig.canvas.draw()
        self.frame_scale.grid(row = 3, column = 0, pady = 2,sticky="N")
        self.message_label.config(text="Please select a frame where the slit shows do to not above 110")
        self.select_frame_button.grid(row = 4, column = 0, pady = 10)

    def create_canvas(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.frame_scale = tk.Scale(self.root, from_=0, to=127, orient=tk.HORIZONTAL, length=400)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0,padx=0, pady=0, sticky="NSWE")
        self.frame_scale.configure(command=self.update_image)

    def create_new_canvas(self):
        self.fig, (self.ax, self.ax1) = plt.subplots(nrows=1, ncols=2, constrained_layout = True)
        self.ax.set_axis_off()
        self.ax1.set_axis_off()
        self.ax.imshow(self.data[0], cmap='gray')
        self.ax1.imshow(self.new_data[0], cmap='gray')
        self.frame_scale = tk.Scale(self.root, from_=0, to=127, orient=tk.HORIZONTAL, length=800)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0,padx=0, pady=0, sticky="NSWE")
        self.frame_scale.configure(command=self.update_image)

    def select_slit_area(self):
        self.message_label.config(text="Please select slit area",font=("Arial", 14))
        self.root.update()
        self.rec = RectangleDrawer(self.fig,self.ax)
        self.select_slit_button_1.grid(row = 5, column = 0, pady = 5)

    def get_ret_vals(self):
        self.slit_area = self.rec.ret_vals()
        print(self.slit_area)
        self.rec.kill()
        self.rec = None
        self.clean_button.grid(row = 4, column = 0, pady = 10)
        self.select_slit_button_1.grid_remove()

    def switch_video(self, value):
        value = int(value)
        self.new_data = self.tresh_videos[value]
        treshes = [i / 1000 for i in range(0, 100, 5)]
        self.message_label.config(text=f"tresh = {treshes[value]}")
        self.update_image(self.current_frame)

    def choose_frame(self):
        self.slit_frame_idx =int(self.frame_scale.get())
        print(self.slit_frame_idx)
        self.slit_frame = self.data[int(self.frame_scale.get())]
        if self.slit_frame_idx >=5 :
            self.data = self.data[self.slit_frame_idx-5:]
            print((self.data).shape)
            self.frame_scale.configure(from_=0,to=127-self.slit_frame_idx+5)

        self.root.update()
        self.select_frame_button.grid_remove()
        self.select_video_button.grid_remove()
        self.select_frame_button.grid_remove()
        self.select_slit_area()

    def update_image(self, value):
        self.current_frame = int(value)
        self.ax.images[0].set_data(self.data[self.current_frame])
        if self.new_data is not None:
            self.ax1.images[0].set_data(self.new_data[self.current_frame])
        self.fig.canvas.draw()

    def first_clean(self):
        self.message_label.config(text="Processing video, please wait...")
        self.root.update()
        # self.new_data = denoise_video(self.data, h=3, template_window_size=7, search_window_size=21)
        # self.new_data = self.data
        self.new_data = denoise_video(self.data)
        self.message_label.config(text="here is you video after first clean")
        self.create_new_canvas()
        self.root.update()
        self.update_image(0)
        self.tresh_button.grid(row = 0, column = 0, pady = 2)

    def show_tresh(self):
        self.clean_button.grid_remove()
        self.tresh_button.grid_remove()
        self.tresh_videos = choose_thresh(self.new_data, self.slit_area)
        self.clean_button.grid_remove()
        self.select_frame_button.grid()
        self.video_scale = tk.Scale(self.root, from_=0, to=19, orient=tk.VERTICAL, length=200, showvalue=False,
                                    command=self.switch_video)
        self.video_scale.grid(row=2,column=0,sticky='W')
        self.current_frame = 0
        self.switch_video(0)

app = App()


