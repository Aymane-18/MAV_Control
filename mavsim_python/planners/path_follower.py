import numpy as np
from math import sin, cos, atan2, pi, sqrt, tan
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap

class PathFollower:
    def __init__(self):
        self.chi_inf = np.radians(50)  # Approach angle for far field
        self.k_path = 0.02  # Tuning parameter for path following
        self.k_orbit = 5.0  # Tuning parameter for orbit following
        self.gravity = 9.81
        self.autopilot_commands = MsgAutopilot()

    def update(self, path: MsgPath, state: MsgState) -> MsgAutopilot:
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        elif path.type == 'helix':
            self._follow_helix(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path: MsgPath, state: MsgState):
        r = path.line_origin
        q = path.line_direction / np.linalg.norm(path.line_direction)
        p = np.array([[state.north, state.east, -state.altitude]]).T  # note inverted altitude
        
        # Calculate the course angle chi_q
        chi_q = atan2(q.item(1), q.item(0))
        chi_q = wrap(chi_q, state.chi)

        # Compute lateral error e_py
        e_py = -sin(chi_q)*(state.north - r.item(0)) + cos(chi_q)*(state.east - r.item(1))

        # Desired course angle (Eq. 10.8 textbook)
        chi_c = chi_q - self.chi_inf * (2/pi) * atan2(self.k_path * e_py, 1)

        # Compute altitude command (Eq. 10.5 textbook)
        n = np.cross(q.flatten(), np.array([0, 0, 1]))
        n /= np.linalg.norm(n[:2])
        s = p - r
        altitude_c = -r.item(2) - sqrt(s.item(0)**2 + s.item(1)**2) * (q.item(2)/sqrt(q.item(0)**2 + q.item(1)**2))

        # Update commands
        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = wrap(chi_c, state.chi)
        self.autopilot_commands.altitude_command = altitude_c
        self.autopilot_commands.phi_feedforward = 0.0  # No roll feedforward for straight line

    def _follow_orbit(self, path: MsgPath, state: MsgState):
        c = path.orbit_center
        rho = path.orbit_radius
        lam = 1 if path.orbit_direction == 'CW' else -1

        d = sqrt((state.north - c.item(0))**2 + (state.east - c.item(1))**2)
        varphi = atan2(state.east - c.item(1), state.north - c.item(0))
        varphi = wrap(varphi, state.chi)

        # Desired course angle (Eq. 10.13 textbook)
        chi_c = varphi + lam * (pi/2 + atan2(self.k_orbit * (d - rho), rho))

        # Altitude command is simply orbit altitude
        altitude_c = -c.item(2)

        # Compute roll feedforward (to improve tracking with wind, Eq. on Slide "Roll Feedforward: wind")
        phi_ff = lam * atan2(state.Vg**2, self.gravity * rho * cos(state.chi - state.psi))

        # Update commands
        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = wrap(chi_c, state.chi)
        self.autopilot_commands.altitude_command = altitude_c
        self.autopilot_commands.phi_feedforward = phi_ff


    def _follow_helix(self, path: MsgPath, state: MsgState):
        c = path.orbit_center
        rho = path.orbit_radius
        lam = 1 if path.orbit_direction == 'CW' else -1
        gamma = path.helix_climb_angle
        start_angle = path.helix_start_angle

        dx = state.north - c.item(0)
        dy = state.east - c.item(1)
        d = sqrt(dx**2 + dy**2)
        varphi = atan2(dy, dx)
        varphi = wrap(varphi, state.chi)

        chi_c = varphi + lam * (pi/2 + atan2(self.k_orbit * (d - rho), rho))
        altitude_c = -c.item(2) + rho * (varphi - start_angle) * tan(gamma)
        phi_ff = lam * atan2(state.Vg**2, self.gravity * rho * cos(state.chi - state.psi))

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = wrap(chi_c, state.chi)
        self.autopilot_commands.altitude_command = altitude_c
        self.autopilot_commands.phi_feedforward = phi_ff
