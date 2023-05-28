import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from region_props import clean_area

class RectangleDrawer:
    def __init__(self, fig, ax):
        self.fig, self.current_ax = fig,ax
        self.x1, self.y1 = 0,0
        self.x2, self.y2 = 400, 250
        # Initialize the RectangleSelector
        self.rs = RectangleSelector( self.current_ax,
            self.line_select_callback,
            useblit=True,
            button=[1, 3],  # Don't use middle button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.connection = plt.connect('key_press_event', self.toggle_selector)

    def line_select_callback(self, eclick, erelease):
        # eclick and erelease are the press and release events
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata


    def ret_vals(self):
        return (np.rint(np.array([self.x1,self.y1,self.x2,self.y2]))).astype(int)

    def kill(self):
        self.rs.set_visible(False)
        self.rs.update()
        self.rs.set_active(False)
        plt.disconnect(self.connection)

    def toggle_selector(self, event):
        pass

class RectangleCleaner:
    def __init__(self, fig, ax, data):
        self.fig, self.current_ax = fig,ax

        self.x1, self.y1 = 0,250
        self.x2, self.y2 = 250 ,400
        # Initialize the RectangleSelector
        self.rs = RectangleSelector( self.current_ax,
            self.line_select_callback,
            useblit=True,
            button=[1, 3],  # Don't use middle button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.connection = plt.connect('key_press_event', self.toggle_selector)
        # plt.show()
    def line_select_callback(self, eclick, erelease):
        # eclick and erelease are the press and release events
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        clean_area(self.data,(np.rint(np.array([self.x1,self.y1,self.x2,self.y2]))).astype(int))


    def ret_vals(self):
        return (np.rint(np.array([self.x1,self.y1,self.x2,self.y2]))).astype(int)

    def kill(self):
        self.rs.set_visible(False)
        self.rs.update()
        self.rs.set_active(False)
        plt.disconnect(self.connection)

    def toggle_selector(self, event):
        pass

