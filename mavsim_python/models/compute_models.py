"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('/Users/aymanesaissi/Desktop/mavsim_public-main/mavsim_python/models/model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()

def compute_tf_model(mav, trim_state, trim_input):
    # Set trim values
    mav._state = trim_state
    mav._update_velocity_data()
    
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    # Compute stability derivatives
    a_phi1 = -0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_p * MAV.b / (2 * Va_trim)
    a_phi2 = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a 

    a_theta1 = -MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_q / (2 * MAV.Jy) * ( MAV.c / (2 * Va_trim))
    a_theta2 = -MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_alpha / (2 * MAV.Jy)
    a_theta3 = MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_delta_e / (2 * MAV.Jy)

    # Compute transfer function coefficients for velocity
    dT_dVa_val = dT_dVa(mav, Va_trim, trim_input.throttle)
    dT_ddelta_t_val = dT_ddelta_t(mav, Va_trim, trim_input.throttle)

    a_V1 = (MAV.rho * Va_trim * MAV.S_wing / MAV.mass) * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * trim_input.elevator) - dT_dVa_val / MAV.mass
    a_V2 = dT_ddelta_t_val / MAV.mass
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3



def compute_ss_model(mav, trim_state, trim_input):
    """
    Compute the state-space model for the MAV around the trim conditions.

    Parameters:
    mav: MAV object containing aircraft parameters and methods
    trim_state: Trimmed state vector
    trim_input: Trimmed input vector

    Returns:
    A_lon: Longitudinal state matrix
    B_lon: Longitudinal input matrix
    A_lat: Lateral state matrix
    B_lat: Lateral input matrix
    """

    # Convert trim state to Euler state
    x_euler = euler_state(trim_state)

    # Compute Jacobian matrices A and B
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)

    # Define indices for longitudinal and lateral states and inputs
    # These indices should correspond to the specific structure of your state and input vectors
    # Example indices:
    # Longitudinal states: [u, w, q, theta, h]
    lon_states = [3, 5, 10, 7, 2]
    # Longitudinal inputs: [elevator, throttle]
    lon_inputs = [0, 3]
    # Lateral states: [v, p, r, phi, psi]
    lat_states = [4, 9, 11, 6, 8]
    # Lateral inputs: [aileron, rudder]
    lat_inputs = [1, 2]

    # Extract longitudinal A and B matrices
    A_lon = A[np.ix_(lon_states, lon_states)]
    B_lon = B[np.ix_(lon_states, lon_inputs)]

    # Extract lateral A and B matrices
    A_lat = A[np.ix_(lat_states, lat_states)]
    B_lat = B[np.ix_(lat_states, lat_inputs)]

    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    """
    Converts quaternion-based state to Euler-based state.
    """
    x_euler = np.copy(x_quat[:12])  # First 12 elements (excluding quaternion)
    phi, theta, psi = quaternion_to_euler(x_quat[6:10])  # Convert quaternion to Euler angles
    x_euler[6:9] = np.array([[phi], [theta], [psi]])  # Replace quaternion with Euler angles
    return x_euler



def quaternion_state(x_euler):
    """
    Converts Euler-based state to quaternion-based state.
    """
    x_quat = np.zeros((13, 1))  # 13-element state vector
    x_quat[:6] = x_euler[:6]  # Copy first 6 elements (position, velocity)
    quat = Euler2Quaternion(x_euler[6], x_euler[7], x_euler[8])  # Convert Euler to quaternion
    x_quat[6:10] = quat.reshape((4, 1))  # Store quaternion values
    x_quat[10:13] = x_euler[9:12]  # Copy angular velocities (p, q, r)
    return x_quat


def f_euler(mav, x_euler, delta):
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    
    forces_moments = mav._forces_moments(delta)
    f_quat = mav._f(x_quat[0:13], forces_moments)

    # Convert quaternion derivative to Euler angle derivatives
    phi, theta, psi = quaternion_to_euler(x_quat[6:10])
    phi_dot = x_quat[10] + np.sin(phi) * np.tan(theta) * x_quat[11] + np.cos(phi) * np.tan(theta) * x_quat[12]
    theta_dot = np.cos(phi) * x_quat[11] - np.sin(phi) * x_quat[12]
    psi_dot = np.sin(phi)/np.cos(theta) * x_quat[11] + np.cos(phi)/np.cos(theta) * x_quat[12]

    f_euler_ = np.zeros((12, 1))
    f_euler_[0:3] = f_quat[0:3]       # pn_dot, pe_dot, pd_dot
    f_euler_[3:6] = f_quat[3:6]       # u_dot, v_dot, w_dot
    f_euler_[6] = phi_dot
    f_euler_[7] = theta_dot
    f_euler_[8] = psi_dot
    f_euler_[9:12] = f_quat[10:13]    # p_dot, q_dot, r_dot

    return f_euler_

def df_dx(mav, x_euler, delta):
    eps = 0.01
    A = np.zeros((12, 12))
    f_x = f_euler(mav, x_euler, delta)
    
    for i in range(12):
        x_eps = np.copy(x_euler)
        x_eps[i] += eps
        f_x_eps = f_euler(mav, x_eps, delta)
        A[:, i] = (f_x_eps - f_x).flatten() / eps

    return A


def df_du(mav, x_euler, delta):
    """
    Computes the Jacobian matrix B = df/du numerically.
    """
    eps = 0.01
    B = np.zeros((12, 4))  # Jacobian of f wrt control inputs

    # Convert delta to a NumPy array
    delta_array = np.array([delta.elevator, delta.aileron, delta.rudder, delta.throttle])

    # Compute f_x for the original state
    f_x = f_euler(mav, x_euler, delta)

    # Loop over each control input
    for i in range(4):
        delta_eps = np.copy(delta_array)  # Copy as NumPy array
        delta_eps[i] += eps  # Apply small perturbation

        # Convert back to MsgDelta object for mav input
        delta_eps_obj = MsgDelta(elevator=delta_eps[0], aileron=delta_eps[1], 
                                 rudder=delta_eps[2], throttle=delta_eps[3])

        # Compute new f_x after perturbation
        f_x_eps = f_euler(mav, x_euler, delta_eps_obj)

        # Compute numerical derivative
        B[:, i] = (f_x_eps - f_x).flatten() / eps

    return B


def dT_dVa(mav, Va, delta_t):
    """
    Returns the derivative of motor thrust with respect to airspeed Va.
    """
    eps = 0.01  # Small perturbation in airspeed

    # Compute thrust for perturbed and unperturbed airspeeds
    T1, _ = mav._motor_thrust_torque(Va + eps, delta_t)  # Perturbed
    T0, _ = mav._motor_thrust_torque(Va, delta_t)  # Baseline

    # Numerical derivative approximation
    dT_dVa = (T1 - T0) / eps

    return dT_dVa




def dT_ddelta_t(mav, Va, delta_t):
    """
    Returns the derivative of motor thrust with respect to throttle (delta_t).
    """
    eps = 0.01  # Small perturbation in throttle

    # Compute thrust for perturbed and unperturbed throttle
    T1, _ = mav._motor_thrust_torque(Va, delta_t + eps)  # Perturbed throttle
    T0, _ = mav._motor_thrust_torque(Va, delta_t)  # Baseline

    # Numerical derivative approximation
    dT_ddelta_t = (T1 - T0) / eps

    return dT_ddelta_t

