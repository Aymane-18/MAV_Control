import numpy as np
import models.model_coef as TF
import parameters.aerosonde_parameters as MAV

# constants
gravity = MAV.gravity
Va0 = TF.Va_trim  # trim airspeed
rho = MAV.rho     # air density
sigma = 0.05      # for differentiator

#----------roll loop-------------
zeta_roll = 0.707
wn_roll = np.sqrt(abs(TF.a_phi2)) * 4.0  # tuning factor ~3.0

roll_kp = wn_roll**2 / TF.a_phi2
roll_kd = (2.0 * zeta_roll * wn_roll - TF.a_phi1) / TF.a_phi2

#----------course loop-------------
zeta_course = 0.707
wn_course = wn_roll / 30.0  # slower than roll loop

course_kp = 2.0 * zeta_course * wn_course * Va0 / gravity
course_ki = wn_course**2 * Va0 / gravity

#----------yaw damper-------------
yaw_damper_p_wo = 1.0
yaw_damper_kr = 0.5

#----------pitch loop-------------
zeta_pitch = 0.707
wn_pitch = np.sqrt(abs(TF.a_theta3)) * 3.0  # similar to roll loop

pitch_kp = (wn_pitch**2 - TF.a_theta2) / TF.a_theta3
pitch_kd = (2.0 * zeta_pitch * wn_pitch - TF.a_theta1) / TF.a_theta3
K_theta_DC = pitch_kp * TF.a_theta3 / (wn_pitch**2)

#----------altitude loop-------------
zeta_altitude = 0.707
wn_altitude = wn_pitch / 20.0

altitude_kp = 2.0 * zeta_altitude * wn_altitude / K_theta_DC
altitude_ki = wn_altitude**2 / K_theta_DC
altitude_zone = 10.0  # meters (range around commanded altitude for switching)

#----------airspeed hold using throttle-------------
zeta_airspeed_throttle = 0.707
wn_airspeed_throttle = 0.5  # tuning parameter (start around 0.5)

airspeed_throttle_kp = (2.0 * zeta_airspeed_throttle * wn_airspeed_throttle - TF.a_V1) / TF.a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2 / TF.a_V2
