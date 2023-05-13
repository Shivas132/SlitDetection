import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


class RectangleDrawer:
    def __init__(self, fig, ax):
        self.fig, self.current_ax = fig,ax
        # self.N = 100000  # If N is large one can see improvement by using blitting!
        # self.x = np.linspace(0.0, 10.0, self.N)  # Create x values for the plot
        # self.plot_data()  # Plot the initial data
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

    # def plot_data(self):
    #     # Plot something
    #     self.current_ax.plot(self.x, +np.sin(.2*np.pi*self.x), lw=3.5, c='b', alpha=.7)
    #     self.current_ax.plot(self.x, +np.cos(.2*np.pi*self.x), lw=3.5, c='r', alpha=.5)
    #     self.current_ax.plot(self.x, -np.sin(.2*np.pi*self.x), lw=3.5, c='g', alpha=.3)

    def line_select_callback(self, eclick, erelease):
        # eclick and erelease are the press and release events
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (self.x1, self.y1, self.x2, self.y2))
        # print(" The button you used was: %s %s" % (eclick.button, erelease.button))
        # return self.x1,self.y1,self.x2,self.y2

    def ret_vals(self):
        return (np.rint(np.array([self.x1,self.y1,self.x2,self.y2]))).astype(int)

    def kill(self):
        self.rs.set_visible(False)
        self.rs.update()
        self.rs.set_active(False)
        plt.disconnect(self.connection)

    def toggle_selector(self, event):
        pass

