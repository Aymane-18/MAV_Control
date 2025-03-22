"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/10/22 - RWB
"""
import numpy as np
from numpy import array, sin, cos, radians, concatenate, zeros, diag
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
        # initialize integrators and delay variables
        self.integratorCourse = 0
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.errorCourseD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0

        # compute LQR gains
        CrLat = array([[0, 0, 0, 0, 0, 1]])  # integral of chi
        AAlat = concatenate((
            concatenate((M.A_lat, zeros((5,1))), axis=1),
            concatenate((CrLat, zeros((1,1))), axis=1)), axis=0)
        BBlat = concatenate((M.B_lat, zeros((1,2))), axis=0)
        Qlat = diag([1, 1, 1, 1, 10, 10])  # v, p, r, phi, chi, intChi
        Rlat = diag([1, 1])  # aileron, rudder
        Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
        self.Klat = inv(Rlat) @ BBlat.T @ Plat

        CrLon = array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        AAlon = concatenate((
            concatenate((M.A_lon, zeros((5,2))), axis=1),
            concatenate((CrLon, zeros((2,2))), axis=1)), axis=0)
        BBlon = concatenate((M.B_lon, zeros((2, 2))), axis=0)
        Qlon = diag([10, 10, 10, 1, 10, 1, 1])  # u, w, q, theta, h, intH, intVa
        Rlon = diag([1, 1])  # elevator, throttle
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        self.Klon = inv(Rlon) @ BBlon.T @ Plon

        self.commanded_state = MsgState()

    def update(self, cmd, state):
        # ----------------- LATERAL ------------------ #
        chi_c = wrap(cmd.course_command, state.chi)
        chi_error = chi_c - state.chi
        self.integratorCourse += (self.Ts/2.0) * (chi_error + self.errorCourseD1)
        self.errorCourseD1 = chi_error
        
        xLat = array([[state.v],
                      [state.p],
                      [state.r],
                      [state.phi],
                      [state.chi],
                      [self.integratorCourse]])
        
        delta_lat = -self.Klat @ xLat
        delta_a = saturate(delta_lat.item(0), -1, 1)
        delta_r = saturate(delta_lat.item(1), -1, 1)

        # ----------------- LONGITUDINAL ------------------ #
        h_c = saturate(cmd.altitude_command,
                      state.altitude - AP.altitude_zone,
                      state.altitude + AP.altitude_zone)
        h_error = h_c - state.altitude
        self.integratorAltitude += (self.Ts/2.0) * (h_error + self.errorAltitudeD1)
        self.errorAltitudeD1 = h_error

        Va_error = cmd.airspeed_command - state.Va
        self.integratorAirspeed += (self.Ts/2.0) * (Va_error + self.errorAirspeedD1)
        self.errorAirspeedD1 = Va_error

        xLon = array([[state.u],
                      [state.w],
                      [state.q],
                      [state.theta],
                      [state.altitude],
                      [self.integratorAltitude],
                      [self.integratorAirspeed]])

        delta_lon = -self.Klon @ xLon
        delta_e = saturate(delta_lon.item(0), -1, 1)
        delta_t = saturate(delta_lon.item(1), 0.0, 1.0)

        # construct control outputs and commanded states
        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)

        self.commanded_state.altitude = h_c
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = state.phi
        self.commanded_state.theta = state.theta
        self.commanded_state.chi = chi_c

        return delta, self.commanded_state
