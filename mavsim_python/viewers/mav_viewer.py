"""
mavsim_python: mav viewer (for chapter 2)
    - Beard & McLain, PUP, 2012
    - Update history:
        1/15/2019 - RWB
        4/15/2019 - BGM
        3/31/2020 - RWB
        7/13/2023 - RWB
        3/25/2024 - Carson Moon
"""
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector
# from viewers.draw_mav import DrawMav
from viewers.draw_mav_stl import DrawMav
from time import time
import numpy as np

class MavViewer():
    def __init__(self, app, ts_refresh=1./30.):
        self.scale = 100
        # initialize Qt gui application and window
        self.app = app  # initialize QT, external so that only one QT process is running
        self.window = gl.GLViewWidget()  # initialize the view object
        #gl.GLViewWidget.getViewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.window.setWindowTitle('MAV Viewer')
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(300, 300, 300) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer

        # Add XYZ Axes (X, Y, Z)
        self.add_xyz_axes()
        
        self.window.setCameraPosition(distance=200) # distance from center of plot to camera
        self.window.setBackgroundColor((61, 112, 143))  # set background color to black
        self.window.setGeometry(0, 0, 750, 750)  # args: upper_left_x, upper_right_y, width, height
        # center = self.window.cameraPosition()
        # center.setX(250)
        # center.setY(250)
        # center.setZ(0)
        # self.window.setCameraPosition(pos=center, distance=self.scale, elevation=50, azimuth=-90)
#        self.window.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front

        self.plot_initialized = False # has the mav been plotted yet?
        self.mav_plot = []
        self.ts_refresh = ts_refresh
        self.t = time()
        self.t_next = self.t    

    def add_xyz_axes(self):
        """ Function to add XYZ axis lines in the 3D viewer. """
        
        # Create X, Y, Z axis lines using GLLinePlotItem
        # Each axis will be a line going from (0, 0, 0) to (some extent in each axis)
        axis_length = 100  # Length of each axis

        # X Axis (Red)
        x_axis = np.array([[0, 0, 0], [axis_length, 0, 0]])  # Line from origin to (100, 0, 0)
        x_axis_item = gl.GLLinePlotItem(pos=x_axis, color=(1, 0, 0, 1), width=2)  # Red axis
        self.window.addItem(x_axis_item)

        # Y Axis (Green)
        y_axis = np.array([[0, 0, 0], [0, axis_length, 0]])  # Line from origin to (0, 100, 0)
        y_axis_item = gl.GLLinePlotItem(pos=y_axis, color=(0, 1, 0, 1), width=2)  # Green axis
        self.window.addItem(y_axis_item)

        # Z Axis (Blue)
        z_axis = np.array([[0, 0, 0], [0, 0, axis_length]])  # Line from origin to (0, 0, 100)
        z_axis_item = gl.GLLinePlotItem(pos=z_axis, color=(0, 0, 1, 1), width=2)  # Blue axis
        self.window.addItem(z_axis_item)

        # Optionally, you can label the axes as well if desired
        # You can create text labels at appropriate positions to indicate the axis direction
        # This part is not mandatory but can help in identifying the axes
    def update(self, state):
        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            self.mav_plot = DrawMav(state, self.window)
            self.plot_initialized = True
        # else update drawing on all other calls to update()
        else:
            t = time()
            if t-self.t_next > 0.0:
                self.mav_plot.update(state)
                self.t = t
                self.t_next = t + self.ts_refresh
        # update the center of the camera view to the mav location
        view_location = Vector(state.east, state.north, state.altitude)  # defined in ENU coordinates
        self.window.opts['center'] = view_location
        # redraw
    
    def process_app(self):
        self.app.processEvents()

    def clear_viewer(self):
        self.window.clear()