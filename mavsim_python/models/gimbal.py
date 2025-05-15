import numpy as np
from tools.rotations import euler_to_rotation
import parameters.camera_parameters as CAM

class Gimbal:
    def pointAtGround(self, mav):
        az_d = 0.0
        el_d = np.radians(-90)
        u_az = CAM.k_az * (az_d - mav.camera_az)
        u_el = CAM.k_el * (el_d - mav.camera_el)
        return np.array([[u_az], [u_el]])

    def pointAtPosition(self, mav, target_position):
        # MAV inertial position
        mav_position = np.array([[mav.north], [mav.east], [-mav.altitude]])

        # Line-of-sight vector in inertial frame
        ell_i = target_position - mav_position

        # Rotate LOS vector into body frame and normalize
        R_v_b = euler_to_rotation(mav.phi, mav.theta, mav.psi).T
        ell_b = R_v_b @ ell_i
        ell_b = ell_b / np.linalg.norm(ell_b)

        # Use camera_az and camera_el instead of gimbal_az and gimbal_el
        return self.pointAlongVector(ell_b, mav.camera_az, mav.camera_el)

    def pointAlongVector(self, ell, azimuth, elevation):
        # Compute desired azimuth and elevation from LOS vector
        az_d = np.arctan2(ell[1, 0], ell[0, 0])
        el_d = np.arctan2(-ell[2, 0], np.sqrt(ell[0, 0]**2 + ell[1, 0]**2))

        # Proportional control
        u_az = CAM.k_az * (az_d - azimuth)
        u_el = CAM.k_el * (el_d - elevation)

        return np.array([[u_az], [u_el]])
