"""
mavsim_python: world viewer (for chapter 12)
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - RWB
        3/30/2022 - RWB
"""
import numpy as np
from viewers.draw_mav_stl import DrawMav
from viewers.draw_path import DrawPath
from viewers.draw_waypoints import DrawWaypoints
from viewers.draw_map import DrawMap
from viewers.draw_target import DrawTarget
from viewers.draw_camera_fov import DrawFov
import pyqtgraph.opengl as gl

class MAVWorldCameraViewer:
    def __init__(self,app, plot_path=True):
        self.scale = 2500
        # initialize Qt gui application and window
        self.app = app  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('World Viewer')
        self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(self.scale/20, self.scale/20, self.scale/20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        center = self.window.cameraPosition()
        center.setX(1000)
        center.setY(1000)
        center.setZ(0)
        self.window.setCameraPosition(pos=center, distance=self.scale, elevation=50, azimuth=-90)
        self.window.setBackgroundColor((61, 112, 143))  # set background color to black
        # self.window.resize(*(4000, 4000))  # not sure how to resize window
        self.window.show()  # display configured window
        self.window.raise_()  # bring window to the front
        self.plot_initialized = False  # has the mav been plotted yet?
        self.mav_plot = []
        self.fov_plot = []
        self.path_plot = []
        self.waypoint_plot = []
        self.map_plot = []
        self.target_plot = []
        self.flag_plot_path = plot_path

    def update(self, state, target_position, path, waypoints, map):
        blue = np.array([[30, 144, 255, 255]])/255.
        red = np.array([[1., 0., 0., 1]])
        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            self.map_plot = DrawMap(map, self.window)
            self.path_plot = DrawPath(path, red, self.window)
            if self.flag_plot_path is True:
                self.waypoint_plot = DrawWaypoints(waypoints, path.orbit_radius, blue, self.window)
            self.mav_plot = DrawMav(state, self.window)
            self.fov_plot = DrawFov(state, self.window)
            self.target_plot = DrawTarget(target_position, self.window)
            self.plot_initialized = True
            path.plot_updated = True
            waypoints.plot_updated = True
        # else update drawing on all other calls to update()
        else:
            self.mav_plot.update(state)
            self.fov_plot.update(state)
            self.target_plot.update(target_position)
            if self.flag_plot_path is True:
                if not waypoints.plot_updated:
                    self.waypoint_plot.update(waypoints)
                    waypoints.plot_updated = True
                if not path.plot_updated:
                    self.path_plot.update(path, red)
                    path.plot_updated = True
            else:
                self.path_plot.update(path, red)
        # redraw
        self.app.processEvents()
