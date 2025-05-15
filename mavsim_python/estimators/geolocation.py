"""
target geolocation algorithm
    - Beard & McLain, PUP, 2012
    - Updated:
        4/1/2022 - RWB
        4/6/2022 - RWB
        7/13/2023 - RWB
        4/7/2025 - TWM
"""
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.camera_parameters as CAM
from tools.rotations import euler_to_rotation
from estimators.filters import ExtendedKalmanFilterContinuousDiscrete

class Geolocation:
    def __init__(self, ts: float=0.01):
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f, 
            Q=0.01 * np.diag([
                1**2,   # target north position
                1**2,   # target east position
                1**2,   # target down position
                10**2,  # target north velocity
                10**2,  # target east velocity
                10**2,  # target down velocity
                3**2,   # distance to target L
            ]),
            P0=0.1 * np.diag([
                10**2,  # target north position
                10**2,  # target east position
                10**2,  # target down position
                10**2,  # target north velocity
                10**2,  # target east velocity
                10**2,  # target down velocity
                10**2,  # distance to target L
            ]),
            xhat0=np.array([[0., 0., 0., 0., 0., 0., 100.]]).T,  # initial guess
            Qu=0.01 * np.diag([
                1**2, 1**2, 1**2,  # mav position
                1**2, 1**2, 1**2   # mav velocity
            ]),
            Ts=ts,
            N=10
        )
        self.R = 0.1 * np.diag([1.0, 1.0, 1.0, 1.0])  # measurement noise

    def update(self, mav, pixels):
        # Build input vector u from MAV position and velocity
        u = np.array([
            [mav.north],
            [mav.east],
            [-mav.altitude],
            [mav.Va * np.cos(mav.chi)],
            [mav.Va * np.sin(mav.chi)],
            [0.0]  # flat earth assumption: vertical velocity = 0
        ])
        
        # EKF propagate
        xhat, P = self.ekf.propagate_model(u)

        # EKF update using pixel measurements
        y = self.process_measurements(mav, pixels)
        xhat, P = self.ekf.measurement_update(
            y=y, 
            u=u,
            h=self.h,
            R=self.R
        )
        return xhat[0:3, :]  # return target NED position

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # State x = [p_obj (3), v_obj (3), L (1)]
        p_obj = x[0:3]
        v_obj = x[3:6]
        L     = x[6, 0]

        # MAV input u = [p_mav (3), v_mav (3)]
        p_mav = u[0:3]
        v_mav = u[3:6]

        # Compute L_dot using slide 19 equation
        rel = p_obj - p_mav
        L_dot = - (rel.T @ v_mav) / L

        # Constant velocity model
        target_position_dot = v_obj
        target_velocity_dot = np.zeros((3, 1))

        xdot = np.concatenate((target_position_dot, target_velocity_dot, L_dot.reshape(1,1)), axis=0)
        return xdot

    def h(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Measurement model: returns predicted measurement y = [target position; L]
        target_position = x[0:3]
        L = x[6:7]
        y = np.concatenate((target_position, L), axis=0)
        return y

    def process_measurements(self, mav, pixels):
        # Step 1: Get MAV position
        h = mav.altitude
        mav_position = np.array([[mav.north], [mav.east], [-h]])

        # Step 2: Line of sight vector in camera frame
        ell = np.array([[pixels.pixel_x], [pixels.pixel_y], [CAM.f]])
        ell_c = ell / np.linalg.norm(ell)

        # Step 3: Compute total rotation from camera to inertial frame
        R_b_i = euler_to_rotation(mav.phi, mav.theta, mav.psi).T  # inertial <- body

        # Use camera_az and camera_el instead of gimbal_az and gimbal_el
        R_g_b = np.array([
            [np.cos(mav.camera_az), -np.sin(mav.camera_az), 0],
            [np.sin(mav.camera_az),  np.cos(mav.camera_az), 0],
            [0,                     0,                      1]
        ]) @ np.array([
            [np.cos(mav.camera_el), 0, np.sin(mav.camera_el)],
            [0,                    1, 0],
            [-np.sin(mav.camera_el), 0, np.cos(mav.camera_el)]
        ])  # body <- gimbal

        R_c_g = np.eye(3)  # camera and gimbal assumed aligned

        # Step 4: Rotate LOS vector to inertial frame
        ell_i = R_b_i @ R_g_b @ R_c_g @ ell_c

        # Step 5: Estimate range L using flat-earth assumption
        L = h / ell_i[2, 0]

        # Step 6: Compute target position
        target_position = mav_position + L * ell_i

        # Step 7: Return measurement y = [target position; L]
        y = np.concatenate((target_position, np.array([[L]])), axis=0)
        return y
