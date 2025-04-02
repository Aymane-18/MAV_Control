"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        """
        Update the airspeed, angle of attack, and sideslip angle based on wind data.
        """
        steady_state = wind[0:3]
        gust = wind[3:6]

        # Convert wind from world to body frame
        wind_body = quaternion_to_rotation(self._state[6:10]).T @ steady_state
        wind_body += gust  # Add gust component

        # Compute velocity relative to airmass
        ur = self._state.item(3) - wind_body.item(0)
        vr = self._state.item(4) - wind_body.item(1)
        wr = self._state.item(5) - wind_body.item(2)

        # Compute airspeed
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)

        # Compute angle of attack α = arctan(wr / ur)
        self._alpha = np.arctan2(wr, ur)

        # Compute sideslip angle β = arcsin(vr / Va)
        self._beta = np.arcsin(vr / self._Va)

        # Update stored wind vector
        self._wind = wind_body


    def _forces_moments(self, delta):
            """
            Computes the aerodynamic, propulsive, and gravitational forces and moments acting on the MAV.
            :param delta: np.array([delta_a, delta_e, delta_r, delta_t]) - control inputs
            :return: Forces and moments on the UAV np.array([Fx, Fy, Fz, Mx, My, Mz])
            """

            # Extract states
            phi, theta, psi = quaternion_to_euler(self._state[6:10])
            e0, e1, e2, e3 = self._state[6:10, 0]
            p = self._state.item(10)  # roll rate
            q = self._state.item(11)  # pitch rate
            r = self._state.item(12)  # yaw rate

            # Compute dynamic pressure
            q_bar = 0.5 * MAV.rho * self._Va ** 2

            # Compute gravity force in body frame
            # -- Gravity forces
            Fgx = MAV.mass * MAV.gravity * (2 * (e1 * e3 - e2 * e0))
            Fgy = MAV.mass * MAV.gravity * (2 * (e2 * e3 + e1 * e0))
            Fgz = MAV.mass * MAV.gravity * (e3**2 + e0**2 - e1**2 - e2**2)

            # Compute aerodynamic force coefficients using provided summary equations
            CX_alpha = -(MAV.C_D_0 + MAV.C_D_alpha * self._alpha) * np.cos(self._alpha) + (MAV.C_L_0+ MAV.C_L_alpha * self._alpha) * np.sin(self._alpha)
            CX_q = -MAV.C_D_q * np.cos(self._alpha) + MAV.C_L_q * np.sin(self._alpha)
            CX_delta_e = -MAV.C_D_delta_e * np.cos(self._alpha) + MAV.C_L_delta_e * np.sin(self._alpha)

            CZ_alpha = -(MAV.C_D_0 + MAV.C_D_alpha * self._alpha)*np.sin(self._alpha) - (MAV.C_L_0+ MAV.C_L_alpha * self._alpha) * np.cos(self._alpha)
            CZ_q = -MAV.C_D_q * np.sin(self._alpha) - MAV.C_L_q * np.cos(self._alpha)
            CZ_delta_e = -MAV.C_D_delta_e * np.sin(self._alpha) - MAV.C_L_delta_e * np.cos(self._alpha)

            # Compute aerodynamic forces
            
            fx_aero = q_bar * MAV.S_wing * (CX_alpha + CX_q * (MAV.c / (2 * self._Va)) * q + CX_delta_e * delta.elevator)



            fy = Fgy + q_bar * MAV.S_wing * (
                MAV.C_Y_0 + MAV.C_Y_beta * self._beta + MAV.C_Y_p * (MAV.b / (2 * self._Va)) * p +
                MAV.C_Y_r * (MAV.b / (2 * self._Va)) * r + MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder
            )
            fz = Fgz + q_bar * MAV.S_wing * (
                CZ_alpha + CZ_q * (MAV.c / (2 * self._Va)) * q + CZ_delta_e * delta.elevator
            )

            # Compute aerodynamic moments
            Mx = q_bar * MAV.S_wing * MAV.b * (
                MAV.C_ell_0 + MAV.C_ell_beta * self._beta + MAV.C_ell_p * (MAV.b / (2 * self._Va)) * p +
                MAV.C_ell_r * (MAV.b / (2 * self._Va)) * r + MAV.C_ell_delta_a * delta.aileron + MAV.C_ell_delta_r * delta.rudder
            )

            My = q_bar * MAV.S_wing * MAV.c * (
                MAV.C_m_0 + MAV.C_m_alpha * self._alpha + MAV.C_m_q * (MAV.c / (2 * self._Va)) * q + MAV.C_m_delta_e * delta.elevator
            )

            Mz = q_bar * MAV.S_wing * MAV.b * (
                MAV.C_n_0 + MAV.C_n_beta * self._beta + MAV.C_n_p * (MAV.b / (2 * self._Va)) * p +
                MAV.C_n_r * (MAV.b / (2 * self._Va)) * r + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder
            )

            # Compute propeller thrust and torque
            thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

            # Add propeller effects
            fx =  Fgx + fx_aero + thrust_prop  # Add thrust
            Mx = Mx - torque_prop  # Propeller torque affects yaw


            # Return forces and moments
            forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
            return forces_moments



    def _motor_thrust_torque(self, Va, delta_t):
        """
        Compute the thrust and torque produced by the propeller.
        :param Va: Airspeed
        :param delta_t: Throttle command (0 to 1)
        :return: (Thrust, Torque)
        """
        # Compute motor input voltage
        v_in = MAV.V_max * delta_t

        # Compute propeller angular speed (Omega)
        a = MAV.rho * (MAV.D_prop ** 5) / (2 * np.pi) ** 2 * MAV.C_Q0
        b = MAV.rho * (MAV.D_prop ** 4) / (2 * np.pi) * MAV.C_Q1 * Va + (MAV.KQ * MAV.KV / MAV.R_motor)
        c = MAV.rho * (MAV.D_prop ** 3) * MAV.C_Q2 * Va ** 2 - (MAV.KQ / MAV.R_motor) * v_in + MAV.KQ * MAV.i0

        # Solve for Omega using quadratic formula
        Omega_p = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        # Compute advance ratio
        J_op = (2 * np.pi * Va) / (Omega_p * MAV.D_prop)

        # Compute thrust and torque coefficients
        C_T = MAV.C_T2 * J_op ** 2 + MAV.C_T1 * J_op + MAV.C_T0
        C_Q = MAV.C_Q2 * J_op ** 2 + MAV.C_Q1 * J_op + MAV.C_Q0

        # Compute thrust and torque
        T_p = MAV.rho * (Omega_p / (2 * np.pi)) ** 2 * MAV.D_prop ** 4 * C_T
        Q_p = MAV.rho * (Omega_p / (2 * np.pi)) ** 2 * MAV.D_prop ** 5 * C_Q

        return T_p, Q_p


    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0