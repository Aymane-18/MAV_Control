"""
observer
    - Beard & McLain, PUP, 2012
    - Completed and corrected by ChatGPT 2025
"""
import numpy as np
import parameters.control_parameters as CTRL
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV  
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete

class Observer:
    def __init__(self, ts: float, initial_measurements: MsgSensors = MsgSensors()):
        self.Ts = ts
        self.estimated_state = MsgState()

        # Low-pass filters for 8.1
        alpha = 0.95
        self.lpf_gyro_x = AlphaFilter(alpha=alpha, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=alpha, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=alpha, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=alpha, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=alpha, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=alpha, y0=initial_measurements.accel_z)
        self.lpf_abs = AlphaFilter(alpha=alpha, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=alpha, y0=initial_measurements.diff_pressure)

        # EKF for roll and pitch
        self.attitude_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_attitude,
            Q=np.diag([0.001, 0.001]),
            P0=np.diag([0.1, 0.1]),
            xhat0=np.array([[0.], [0.]]),
            Qu=np.diag([SENSOR.gyro_sigma**2] * 3 + [MAV.rho]),
            Ts=ts,
            N=5
        )

        # EKF for position, heading, wind
        self.position_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_smooth,
            Q=np.diag([0.1] * 7),
            P0=np.diag([1.0] * 7),
            xhat0=np.zeros((7, 1)),
            Qu=np.diag([0.1] * 5),
            Ts=ts,
            N=10
        )

        self.R_accel = np.diag([SENSOR.accel_sigma**2] * 3)
        self.R_pseudo = np.diag([0.1, 0.1])
        self.R_gps = np.diag([
            SENSOR.gps_n_sigma**2,
            SENSOR.gps_e_sigma**2,
            SENSOR.gps_Vg_sigma**2,
            SENSOR.gps_course_sigma**2
        ])

        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, measurement: MsgSensors) -> MsgState:
        # 8.1 Low-pass filtered angular rates
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)

        abs_pressure = self.lpf_abs.update(measurement.abs_pressure)
        diff_pressure = self.lpf_diff.update(measurement.diff_pressure)

        self.estimated_state.altitude = -abs_pressure / (MAV.rho * MAV.gravity)
        self.estimated_state.Va = max(np.sqrt(2 * diff_pressure / MAV.rho), 0.01)

        # 8.2 EKF for phi and theta
        u_att = np.array([
            [self.estimated_state.p],
            [self.estimated_state.q],
            [self.estimated_state.r],
            [self.estimated_state.Va]
        ])
        xhat_att, _ = self.attitude_ekf.propagate_model(u_att)
        y_accel = np.array([
            [self.lpf_accel_x.update(measurement.accel_x)],
            [self.lpf_accel_y.update(measurement.accel_y)],
            [self.lpf_accel_z.update(measurement.accel_z)]
        ])
        xhat_att, _ = self.attitude_ekf.measurement_update(y_accel, u_att, self.h_accel, self.R_accel)
        self.estimated_state.phi = xhat_att[0, 0]
        self.estimated_state.theta = xhat_att[1, 0]

        # 8.3 EKF for position, heading, wind
        u_pos = np.array([
            [self.estimated_state.q],
            [self.estimated_state.r],
            [self.estimated_state.Va],
            [self.estimated_state.phi],
            [self.estimated_state.theta]
        ])
        xhat_pos, _ = self.position_ekf.propagate_model(u_pos)
        xhat_pos, _ = self.position_ekf.measurement_update(np.zeros((2, 1)), u_pos, self.h_pseudo, self.R_pseudo)

        if (measurement.gps_n != self.gps_n_old or
            measurement.gps_e != self.gps_e_old or
            measurement.gps_Vg != self.gps_Vg_old or
            measurement.gps_course != self.gps_course_old):
            y_gps = np.array([
                [measurement.gps_n],
                [measurement.gps_e],
                [measurement.gps_Vg],
                [wrap(measurement.gps_course, xhat_pos[3, 0])]
            ])
            xhat_pos, _ = self.position_ekf.measurement_update(y_gps, u_pos, self.h_gps, self.R_gps)
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

        self.estimated_state.north = xhat_pos[0, 0]
        self.estimated_state.east = xhat_pos[1, 0]
        self.estimated_state.Vg = xhat_pos[2, 0]
        self.estimated_state.chi = xhat_pos[3, 0]
        self.estimated_state.wn = xhat_pos[4, 0]
        self.estimated_state.we = xhat_pos[5, 0]

        # Fix: estimate psi from wind triangle
        Vg = self.estimated_state.Vg
        Va = self.estimated_state.Va
        chi = self.estimated_state.chi
        wn = self.estimated_state.wn
        we = self.estimated_state.we
        Vn = Vg * np.cos(chi)
        Ve = Vg * np.sin(chi)
        psi = np.arctan2(Ve , Vn )



        self.estimated_state.psi = wrap(psi, chi)

        # Improved estimate of alpha and beta using body-frame acceleration
        ax = self.lpf_accel_x.update(measurement.accel_x)
        ay = self.lpf_accel_y.update(measurement.accel_y)
        az = self.lpf_accel_z.update(measurement.accel_z)

        theta = self.estimated_state.theta
        Va = self.estimated_state.Va

        ax_corr = ax - MAV.gravity * np.sin(theta)
        az_corr = az + MAV.gravity * np.cos(theta)

        alpha = np.arctan2(az_corr, ax_corr)
        beta = np.arcsin(ay / Va) if Va > 0 else 0.0

        self.estimated_state.alpha = alpha
        self.estimated_state.beta = beta


        # self.estimated_state.alpha = self.estimated_state.theta
        # self.estimated_state.beta = 0.0

        return self.estimated_state

    def f_attitude(self, x, u):
        phi, theta = x[0, 0], x[1, 0]
        p, q, r, Va = u[0, 0], u[1, 0], u[2, 0], u[3, 0]
        phi_dot = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
        theta_dot = np.cos(phi)*q - np.sin(phi)*r
        return np.array([[phi_dot], [theta_dot]])

    def h_accel(self, x, u):
        phi, theta = x[0, 0], x[1, 0]
        p, q, r, Va = u[0, 0], u[1, 0], u[2, 0], u[3, 0]
        ax = q*r*MAV.Jy/MAV.Jx - r*q*MAV.Jz/MAV.Jx + MAV.gravity*np.sin(theta)
        ay = r*p*MAV.Jz/MAV.Jy - p*r*MAV.Jx/MAV.Jy - MAV.gravity*np.cos(theta)*np.sin(phi)
        az = p*q*MAV.Jx/MAV.Jz - q*p*MAV.Jy/MAV.Jz - MAV.gravity*np.cos(theta)*np.cos(phi)
        return np.array([[ax], [ay], [az]])

    def f_smooth(self, x, u):
        pn, pe, Vg, chi, wn, we, psi = x.flatten()
        q, r, Va, phi, theta = u.flatten()
        Vg_safe = max(Vg, 0.1)
        chi_dot = MAV.gravity / Vg_safe * np.tan(phi) * np.cos(theta)
        pn_dot = Vg * np.cos(chi)
        pe_dot = Vg * np.sin(chi)
        psi_dot = r
        return np.array([[pn_dot], [pe_dot], [0.], [chi_dot], [0.], [0.], [psi_dot]])

    def h_pseudo(self, x, u):
        Vg = x[2, 0]
        chi = x[3, 0]
        wn = x[4, 0]
        we = x[5, 0]
        psi = x[6, 0]
        Va = u[2, 0]
        phi = u[3, 0]
        theta = u[4, 0]
        Vn = Va * np.cos(psi) + wn
        Ve = Va * np.sin(psi) + we
        Vg_calc = np.sqrt(Vn**2 + Ve**2)
        chi_calc = np.arctan2(Ve, Vn)
        return np.array([[Vg_calc], [chi_calc]])

    def h_gps(self, x, u):
        return x[[0, 1, 2, 3], :]  # [pn, pe, Vg, chi]
