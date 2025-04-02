import numpy as np
from numpy import array, sin, cos, concatenate, zeros, diag
from scipy.linalg import solve_continuous_are, inv
import parameters.control_parameters as AP
from tools.wrap import wrap
import models.model_coef as M
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

def saturate(input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output

class Autopilot:
    def __init__(self, ts_control):
        self.Ts = ts_control
        self.integratorCourse = 0
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.errorCourseD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0

        # ---------------- LQR: Lateral ----------------
        CrLat = array([[0, 0, 0, 0, 1.0]])  # integral of course
        AAlat = concatenate((concatenate((M.A_lat, zeros((5,1))), axis=1),
                             concatenate((CrLat, zeros((1,1))), axis=1)), axis=0)
        BBlat = concatenate((M.B_lat, zeros((1,2))), axis=0)
        Qlat = diag([0.001, 0.05, 0.5, 50, 1, 50.0])  # v, p, r, phi, chi, intChi

        Rlat = diag([5, 5])  # aileron, rudder
        Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
        self.Klat = inv(Rlat) @ BBlat.T @ Plat

        # ---------------- LQR: Longitudinal ----------------
        CrLon = array([[0, 0, 0, 0, 1.0],  # integral of altitude
                       [1.0 / AP.Va0, 1.0 / AP.Va0, 0, 0, 0]])  # integral of airspeed
        AAlon = concatenate((concatenate((M.A_lon, zeros((5,2))), axis=1),
                             concatenate((CrLon, zeros((2,2))), axis=1)), axis=0)
        BBlon = concatenate((M.B_lon, zeros((2,2))), axis=0)
        Qlon = diag([10, 10, 0.001, 0.01, 10.0, 100.0, 100.0])  # u, w, q, theta, h, int_h, int_Va

        Rlon = diag([3.0, 3.0])  # elevator, throttle
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        self.Klon = inv(Rlon) @ BBlon.T @ Plon

        self.commanded_state = MsgState()

    def update(self, cmd, state):
        # ---------------- Lateral LQR ----------------
        errorAirspeed = state.Va - cmd.airspeed_command
        wrapped_chi = wrap(cmd.course_command, state.chi)
        errorCourse = state.chi - wrapped_chi
        errorCourse = saturate(errorCourse, -np.radians(15), np.radians(15))

        self.integratorCourse += 0.5 * self.Ts * (errorCourse + self.errorCourseD1)
        self.errorCourseD1 = errorCourse

        xLat = array([
            [errorAirspeed * np.sin(state.beta)],
            [state.p],
            [state.r],
            [state.phi],
            [errorCourse],
            [self.integratorCourse]
        ])
        uLat = -self.Klat @ xLat
        delta_a = saturate(uLat.item(0), -np.radians(30), np.radians(30))
        delta_r = saturate(uLat.item(1), -np.radians(30), np.radians(30))

        # ---------------- Longitudinal LQR ----------------
        altitude_c = saturate(cmd.altitude_command,
                              state.altitude - 0.2 * AP.altitude_zone,
                              state.altitude + 0.2 * AP.altitude_zone)
        errorAltitude = -(state.altitude - altitude_c)
  

        self.integratorAltitude += 0.5 * self.Ts * (errorAltitude + self.errorAltitudeD1)
        self.errorAltitudeD1 = errorAltitude

        self.integratorAirspeed += 0.5 * self.Ts * (errorAirspeed + self.errorAirspeedD1)
        self.errorAirspeedD1 = errorAirspeed

        xLon = array([
            [errorAirspeed * np.cos(state.alpha)],  # approximate u
            [errorAirspeed * np.sin(state.alpha)],  # approximate w
            [state.q],
            [state.theta],
            [errorAltitude],
            [self.integratorAltitude],
            [self.integratorAirspeed]
        ])
        uLon = -self.Klon @ xLon
        delta_e = saturate(uLon.item(0), -np.radians(30), np.radians(30))
        delta_t = saturate(uLon.item(1), 0.0, 1.0)

        # ---------------- Return Results ----------------
        delta = MsgDelta(elevator=delta_e, aileron=delta_a, rudder=delta_r, throttle=delta_t)

        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = state.phi
        self.commanded_state.theta = state.theta
        self.commanded_state.chi = cmd.course_command

        return delta, self.commanded_state
