import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from region_props import clean_area


class RectangleDrawer:
    """
        A class for drawing rectangles on a plot.

        Attributes:
            fig (Figure): The figure object.
            current_ax (Axes): The current axes object.
            x1 (float): The x-coordinate of the starting point of the rectangle.
            y1 (float): The y-coordinate of the starting point of the rectangle.
            x2 (float): The x-coordinate of the ending point of the rectangle.
            y2 (float): The y-coordinate of the ending point of the rectangle.
            rs (RectangleSelector): The RectangleSelector instance for interactive rectangle drawing.
            connection (Connection): The connection object for event handling.

        Methods:
            __init__(self, fig: Figure, ax: Axes) -> None:
                Initialize a RectangleDrawer object.

            line_select_callback(self, eclick, erelease) -> None:
                Callback function for the RectangleSelector.

            ret_vals(self) -> np.array:
                Return the coordinates of the rectangle.

            kill(self) -> None:
                Disable the RectangleSelector and disconnect the event.

            toggle_selector(self, event) -> None:
                Toggle the selector.
        """
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
