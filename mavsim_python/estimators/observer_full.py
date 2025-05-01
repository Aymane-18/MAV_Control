'''
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/4/2019 - RWB
        3/6/2024 - RWB
'''
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_rotation
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete


class Observer:
    def __init__(self, ts):
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f,
            Q=np.diag([
                0.1**2, 0.1**2, 0.1**2,  # pn, pe, pd
                0.5**2, 0.5**2, 0.5**2,  # u, v, w
                np.radians(0.1)**2, np.radians(0.1)**2, np.radians(0.1)**2,  # phi, theta, psi
                np.radians(0.01)**2, np.radians(0.01)**2, np.radians(0.01)**2,  # gyro biases
                0.1**2, 0.1**2  # wn, we
            ]),
            P0=np.diag([
                1**2, 1**2, 1**2,
                1**2, 1**2, 1**2,
                np.radians(5)**2, np.radians(5)**2, np.radians(5)**2,
                np.radians(1)**2, np.radians(1)**2, np.radians(1)**2,
                1**2, 1**2
            ]),
            xhat0=np.array([[MAV.north0, MAV.east0, MAV.down0,
                             MAV.Va0, 0, 0,
                             0, 0, MAV.psi0,
                             0, 0, 0,
                             0, 0]]).T,
            Qu=np.diag([
                SENSOR.gyro_sigma**2,
                SENSOR.gyro_sigma**2,
                SENSOR.gyro_sigma**2,
                SENSOR.accel_sigma**2,
                SENSOR.accel_sigma**2,
                SENSOR.accel_sigma**2
            ]),
            Ts=ts,
            N=10
        )
        self.R_analog = np.diag([
            SENSOR.abs_pres_sigma**2,
            SENSOR.diff_pres_sigma**2,
            (0.01)**2
        ])
        self.R_gps = np.diag([
            SENSOR.gps_n_sigma**2,
            SENSOR.gps_e_sigma**2,
            SENSOR.gps_Vg_sigma**2,
            SENSOR.gps_course_sigma**2
        ])
        self.R_pseudo = np.diag([
            0.01**2,
            0.01**2
        ])
        initial_measurements = MsgSensors()
        self.lpf_gyro_x = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_z)
        self.analog_threshold = stats.chi2.isf(q=0.01, df=3)
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=2)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999
        self.estimated_state = MsgState()
        self.elapsed_time = 0

    def update(self, measurement):
        u = np.array([[measurement.gyro_x, measurement.gyro_y, measurement.gyro_z,
                       measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T
        xhat, P = self.ekf.propagate_model(u)
        y_analog = np.array([[measurement.abs_pressure],
                             [measurement.diff_pressure],
                             [0.0]])
        xhat, P = self.ekf.measurement_update(y_analog, u, self.h_analog, self.R_analog)

        y_pseudo = np.array([[0.0], [0.0]])
        xhat, P = self.ekf.measurement_update(y_pseudo, u, self.h_pseudo, self.R_pseudo)

        if (measurement.gps_n != self.gps_n_old) or (measurement.gps_e != self.gps_e_old) or \
           (measurement.gps_Vg != self.gps_Vg_old) or (measurement.gps_course != self.gps_course_old):
            state = to_MsgState(xhat)
            y_chi = wrap(measurement.gps_course, state.chi)
            y_gps = np.array([[measurement.gps_n], [measurement.gps_e],
                              [measurement.gps_Vg], [y_chi]])
            xhat, P = self.ekf.measurement_update(y_gps, u, self.h_gps, self.R_gps)
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

        self.estimated_state = to_MsgState(xhat)
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) - self.estimated_state.bx
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y) - self.estimated_state.by
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z) - self.estimated_state.bz
        self.elapsed_time += SIM.ts_control
        return self.estimated_state

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pos = x[0:3]
        vel_body = x[3:6]
        euler = x[6:9]
        bias = x[9:12]
        wind = x[12:14]
        R = euler_to_rotation(float(euler[0]), float(euler[1]), float(euler[2]))


        y_gyro = u[0:3] - bias.reshape(3, 1)
        y_accel = u[3:6]

        vel_world = R @ vel_body
        pos_dot = vel_world

        u_b = vel_body[0, 0]
        w_b = vel_body[2, 0]
        accel_x = y_accel[0, 0]
        accel_z = y_accel[2, 0]
        theta = euler[1, 0]

        vel_dot = np.zeros((3, 1))
        vel_dot[0, 0] = accel_x + MAV.gravity * np.sin(theta)
        vel_dot[1, 0] = y_accel[1, 0]  # Approximation
        vel_dot[2, 0] = accel_z - MAV.gravity * np.cos(theta)

        Theta_dot = S(euler) @ y_gyro
        bias_dot = np.zeros((3, 1))
        wind_dot = np.zeros((2, 1))

        xdot = np.concatenate((pos_dot, vel_dot, Theta_dot, bias_dot, wind_dot), axis=0)
        return xdot

    def h_analog(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pd = x[2, 0]
        vel_body = x[3:6]
        wn, we = x[12, 0], x[13, 0]
        Va = np.linalg.norm(vel_body - np.array([[wn], [0], [we]]))
        abs_pres = MAV.rho * MAV.gravity * -pd
        diff_pres = 0.5 * MAV.rho * Va ** 2
        beta = np.arcsin(vel_body[1, 0] / Va)
        return np.array([[abs_pres], [diff_pres], [beta]])

    def h_gps(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pos = x[0:3]
        vel_body = x[3:6]
        psi = x[8, 0]
        wn, we = x[12, 0], x[13, 0]
        R = euler_to_rotation(x[6, 0], x[7, 0], psi)
        vel_inertial = R @ vel_body
        Vg = np.linalg.norm(vel_inertial)
        chi = np.arctan2(vel_inertial[1, 0], vel_inertial[0, 0])
        return np.array([[pos[0, 0]], [pos[1, 0]], [Vg], [chi]])

    def h_pseudo(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        vel_body = x[3:6]
        wn, we = x[12, 0], x[13, 0]
        psi = x[8, 0]
        Va = np.linalg.norm(vel_body)
        wind_x = Va * np.cos(psi) + wn
        wind_y = Va * np.sin(psi) + we
        Vg = np.linalg.norm([wind_x, wind_y])
        return np.array([[wind_x - Vg * np.cos(psi)], [wind_y - Vg * np.sin(psi)]])


def to_MsgState(x: np.ndarray) -> MsgState:
    state = MsgState()
    state.north = x.item(0)
    state.east = x.item(1)
    state.altitude = -x.item(2)
    vel_body = x[3:6]
    state.phi = x.item(6)
    state.theta = x.item(7)
    state.psi = x.item(8)
    state.bx = x.item(9)
    state.by = x.item(10)
    state.bz = x.item(11)
    state.wn = x.item(12)
    state.we = x.item(13)
    R = euler_to_rotation(state.phi, state.theta, state.psi)
    vel_world = R @ vel_body
    wind_world = np.array([[state.wn], [state.we], [0]])
    wind_body = R.T @ wind_world
    vel_rel = vel_body - wind_body
    state.Va = np.linalg.norm(vel_rel)
    state.alpha = np.arctan2(vel_rel.item(2), vel_rel.item(0))
    state.beta = np.arcsin(vel_rel.item(1) / state.Va)
    state.Vg = np.linalg.norm(vel_world)
    state.chi = np.arctan2(vel_world.item(1), vel_world.item(0))
    return state


def cross(vec: np.ndarray) -> np.ndarray:
    return np.array([[0, -vec.item(2), vec.item(1)],
                     [vec.item(2), 0, -vec.item(0)],
                     [-vec.item(1), vec.item(0), 0]])


def S(Theta: np.ndarray) -> np.ndarray:
    return np.array([[1, np.sin(Theta.item(0)) * np.tan(Theta.item(1)), np.cos(Theta.item(0)) * np.tan(Theta.item(1))],
                     [0, np.cos(Theta.item(0)), -np.sin(Theta.item(0))],
                     [0, np.sin(Theta.item(0)) / np.cos(Theta.item(1)), np.cos(Theta.item(0)) / np.cos(Theta.item(1))]])
