"""
autopilot block for mavsim_python - Total Energy Control System
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/14/2020 - RWB
"""
import numpy as np
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kr, 0]]),
                        den=np.array([[1, AP.yaw_damper_p_wo]]),
                        Ts=ts_control)

        # instantiate TECS controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        # throttle gains (unitless)
        self.E_kp = 0
        self.E_ki = 0
        # pitch gains
        self.L_kp = 0
        self.L_ki = 0
        # saturated altitude error
        self.h_error_max = 0  # meters
        self.E_integrator = 0
        self.L_integrator = 0
        self.E_error_d1 = 0
        self.L_error_d1 = 0
        self.delta_t_d1 = 0
        self.theta_c_d1 = 0
        self.theta_c_max = 0
        self.Ts = ts_control
        self.commanded_state = MsgState()

def update(self, cmd, state):
    ### 1. Lateral Autopilot ###
    # Compute desired roll angle (phi_c) from course command (chi_c)
    phi_c = self.course_from_roll.update(cmd.course - state.chi)
    phi_c = self.saturate(phi_c, -np.radians(30), np.radians(30))  # limit to ±30 degrees

    # Compute aileron deflection using roll controller
    delta_a = self.roll_from_aileron.update(phi_c - state.phi, state.p)

    # Compute rudder deflection using yaw damper
    delta_r = self.yaw_damper.update(state.r)

    ### 2. Longitudinal TECS Autopilot ###
    # Compute altitude and airspeed errors
    h_error = self.saturate(cmd.altitude - state.h, -self.h_error_max, self.h_error_max)
    Va_error = cmd.airspeed - state.Va

    # Total Energy (E) error and Load Factor (L) error
    E_error = Va_error + 9.81 * h_error  # Total energy error
    L_error = -h_error  # Load factor error (opposite sign of h_error)

    # Integrate errors for PI control
    self.E_integrator += (self.Ts / 2) * (E_error + self.E_error_d1)
    self.L_integrator += (self.Ts / 2) * (L_error + self.L_error_d1)

    # Compute throttle control (delta_t) using energy balance
    delta_t = self.E_kp * E_error + self.E_ki * self.E_integrator
    delta_t = self.saturate(delta_t, 0, 1)  # limit throttle between 0 and 1

    # Compute pitch angle command (theta_c)
    theta_c = self.L_kp * L_error + self.L_ki * self.L_integrator
    theta_c = self.saturate(theta_c, -self.theta_c_max, self.theta_c_max)

    # Compute elevator deflection (delta_e) using pitch controller
    delta_e = self.pitch_from_elevator.update(theta_c - state.theta, state.q)

    ### Construct Output ###
    delta = MsgDelta(elevator=delta_e, aileron=delta_a, rudder=delta_r, throttle=delta_t)

    # Update commanded state
    self.commanded_state.altitude = cmd.altitude
    self.commanded_state.Va = cmd.airspeed
    self.commanded_state.phi = phi_c
    self.commanded_state.theta = theta_c
    self.commanded_state.chi = cmd.course

    # Store previous errors for next iteration
    self.E_error_d1 = E_error
    self.L_error_d1 = L_error

    return delta, self.commanded_state


def saturate(self, input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output
