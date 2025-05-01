import numpy as np

class MsgPath:
    '''
        Message class that defines a path.
        Supports:
        - 'line': defined by airspeed, line_origin, line_direction
        - 'orbit': defined by orbit_center, orbit_radius, orbit_direction
        - 'helix': adds helix_climb_angle and helix_start_angle
    '''
    def __init__(self,
                 type='line',
                 airspeed=25.0,
                 line_origin=np.array([[0.0, 0.0, 0.0]]).T,
                 line_direction=np.array([[1.0, 0.0, 0.0]]).T,
                 orbit_center=np.array([[0.0, 0.0, 0.0]]).T,
                 orbit_radius=50.0,
                 orbit_direction='CW',
                 helix_climb_angle=0.0,
                 helix_start_angle=0.0,
                 plot_updated=False):
        self.type = type
        self.airspeed = airspeed
        self.line_origin = line_origin
        self.line_direction = line_direction
        self.orbit_center = orbit_center
        self.orbit_radius = orbit_radius
        self.orbit_direction = orbit_direction
        self.helix_climb_angle = helix_climb_angle
        self.helix_start_angle = helix_start_angle
        self.plot_updated = plot_updated
