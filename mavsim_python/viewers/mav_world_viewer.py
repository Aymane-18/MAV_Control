import numpy as np
from viewers.draw_mav_stl import DrawMav
from viewers.draw_path import DrawPath
from viewers.draw_waypoints import DrawWaypoints
from viewers.draw_map import DrawMap
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector

class MAVWorldViewer:
    def __init__(self, app):
        self.scale = 2500
        self.app = app
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('World Viewer')
        self.window.setGeometry(0, 0, 750, 750)
        grid = gl.GLGridItem()
        grid.scale(self.scale/20, self.scale/20, self.scale/20)
        self.window.addItem(grid)
        center = self.window.cameraPosition()
        center.setX(1000)
        center.setY(1000)
        center.setZ(0)
        self.window.setCameraPosition(pos=center, distance=self.scale, elevation=50, azimuth=-90)
        self.window.setBackgroundColor((61, 112, 143))
        self.window.show()
        self.window.raise_()
        self.plot_initialized = False
        self.mav_plot = []
        self.path_plot = []
        self.waypoint_plot = []
        self.map_plot = []

        self.add_xyz_axes()

    def add_xyz_axes(self):
        axis_length = 100
        x_axis = np.array([[0, 0, 0], [axis_length, 0, 0]])
        x_axis_item = gl.GLLinePlotItem(pos=x_axis, color=(1, 0, 0, 1), width=2)
        self.window.addItem(x_axis_item)

        y_axis = np.array([[0, 0, 0], [0, axis_length, 0]])
        y_axis_item = gl.GLLinePlotItem(pos=y_axis, color=(0, 1, 0, 1), width=2)
        self.window.addItem(y_axis_item)

        z_axis = np.array([[0, 0, 0], [0, 0, axis_length]])
        z_axis_item = gl.GLLinePlotItem(pos=z_axis, color=(0, 0, 1, 1), width=2)
        self.window.addItem(z_axis_item)

    def update(self, state, path, waypoints, map):
        blue = np.array([[30, 144, 255, 255]]) / 255.
        red = np.array([[1., 0., 0., 1]])

        if not self.plot_initialized:
            self.map_plot = DrawMap(map, self.window)

            if path is not None:
                self.path_plot = DrawPath(path, red, self.window)
                path.plot_updated = True
            else:
                self.path_plot = None

            if waypoints is not None:
                orbit_radius = getattr(path, 'orbit_radius', 25.0)
                self.waypoint_plot = DrawWaypoints(waypoints, orbit_radius, blue, self.window)
                waypoints.plot_updated = True
            else:
                self.waypoint_plot = None

            self.mav_plot = DrawMav(state, self.window)
            self.plot_initialized = True

        else:
            self.mav_plot.update(state)

            if waypoints is not None and not waypoints.plot_updated:
                self.waypoint_plot.update(waypoints)
                waypoints.plot_updated = True

            if path is not None and not path.plot_updated:
                self.path_plot.update(path, red)
                path.plot_updated = True

        # update the center of the camera view to the mav location
        view_location = Vector(state.east, state.north, state.altitude)
        self.window.opts['center'] = view_location

        self.app.processEvents()

    def clear_viewer(self):
        self.window.clear()
