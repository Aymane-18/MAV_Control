import numpy as np
import pyqtgraph.opengl as gl
from message_types.msg_path import MsgPath

class DrawPath:
    def __init__(self, path: MsgPath, color: np.ndarray, window: gl.GLViewWidget):
        self.color = color
        if path.type == 'line':
            scale = 1000
            points = straight_line_points(path, scale)
        elif path.type == 'orbit':
            points = orbit_points(path)
        elif path.type == 'helix':
            points = helix_points(path)
        else:
            raise ValueError(f"Unsupported path type: {path.type}")

        path_color = np.tile(color, (points.shape[0], 1))
        self.path_plot_object = gl.GLLinePlotItem(pos=points,
                                                  color=path_color,
                                                  width=1,
                                                  antialias=True,
                                                  mode='line_strip')
        window.addItem(self.path_plot_object)

    def update(self, path: MsgPath, color: np.ndarray):
        if path.type == 'line':
            scale = 1000
            points = straight_line_points(path, scale)
        elif path.type == 'orbit':
            points = orbit_points(path)
        elif path.type == 'helix':
            points = helix_points(path)
        else:
            raise ValueError(f"Unsupported path type: {path.type}")

        path_color = np.tile(color, (points.shape[0], 1))
        self.path_plot_object.setData(pos=points, color=path_color)


def straight_line_points(path: MsgPath, scale: float):
    points = np.array([[path.line_origin.item(0),
                        path.line_origin.item(1),
                        path.line_origin.item(2)],
                       [path.line_origin.item(0) + scale * path.line_direction.item(0),
                        path.line_origin.item(1) + scale * path.line_direction.item(1),
                        path.line_origin.item(2) + scale * path.line_direction.item(2)]])
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return points @ R.T


def orbit_points(path: MsgPath, N=100):
    theta_list = np.linspace(0, 2*np.pi, N)
    points = np.array([
        [path.orbit_center.item(0) + path.orbit_radius * np.cos(angle),
         path.orbit_center.item(1) + path.orbit_radius * np.sin(angle),
         path.orbit_center.item(2)] for angle in theta_list
    ])
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return points @ R.T


def helix_points(path: MsgPath, N=300):
    radius = path.orbit_radius
    climb = path.helix_climb_angle
    angle_start = path.helix_start_angle
    num_turns = 3
    theta_list = np.linspace(angle_start, angle_start + num_turns*2*np.pi, N)
    
    points = np.array([
        [path.orbit_center.item(0) + radius * np.cos(theta),
         path.orbit_center.item(1) + radius * np.sin(theta),
         path.orbit_center.item(2) + radius * np.tan(climb) * (theta - angle_start)]
        for theta in theta_list
    ])
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return points @ R.T
