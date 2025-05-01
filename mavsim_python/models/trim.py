"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion
from message_types.msg_delta import MsgDelta
import time

def compute_trim(mav, Va, gamma):
    # Set the initial state with velocity and attitude conditions for straight flight
    e0 = Euler2Quaternion(0., gamma, 0.)
    state0 = np.array([[0],  # pn (North position)
                       [0],  # pe (East position)
                       [0],  # pd (Down position)
                       [Va], # u (Velocity along body x-axis)
                       [0.], # v (Velocity along body y-axis, should be 0 for straight flight)
                       [0.], # w (Velocity along body z-axis)
                       [e0.item(0)], # e0 (Quaternion component)
                       [e0.item(1)], # e1 (Quaternion component)
                       [e0.item(2)], # e2 (Quaternion component)
                       [e0.item(3)], # e3 (Quaternion component)
                       [0.], # p (Roll rate)
                       [0.], # q (Pitch rate)
                       [0.]  # r (Yaw rate)
                       ])
    
    # Initial control inputs (to be optimized)
    delta0 = np.array([[0],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0.5]]) # throttle (assuming mid-power for level flight)
    
    # Combine into optimization variable
    x0 = np.concatenate((state0, delta0), axis=0).flatten()

    # Define equality constraints for trim conditions
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # Ensure airspeed magnitude is Va
                                x[4],  # Force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # Quaternion normalization
                                x[7],  # e1=0 (forces no roll)
                                x[9],  # e3=0 (forces no yaw)
                                x[10],  # p=0 (zero roll rate)
                                x[11],  # q=0 (zero pitch rate)
                                x[12],  # r=0 (zero yaw rate)
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })

    # Solve the minimization problem using SLSQP optimizer
    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    
    # Extract trim state and control input
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))

    # Print results
    trim_input.print()
    print('trim_state=', trim_state.T)

    return trim_state, trim_input



def trim_objective_fun(x, mav, Va, gamma):
    # Extract state and control inputs from x
    state = x[0:13].reshape((-1, 1))
    delta = MsgDelta(elevator=x.item(13),
                     aileron=x.item(14),
                     rudder=x.item(15),
                     throttle=x.item(16))
    
    # Define the desired trim state derivative
    desired_trim_state_dot = np.array([[0., 0., -Va * np.sin(gamma), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
    
    # Update MAV state
    mav._state = state
    mav._update_velocity_data()
    
    # Compute forces and moments
    forces_moments = mav._forces_moments(delta)
    
    # Compute actual state derivative
    actual_state_dot = mav._f(state, forces_moments)
    
    # Compute error between actual and desired state derivative
    error = desired_trim_state_dot - actual_state_dot
    
    # Compute cost function as the squared norm of the error (excluding first two elements)
    J = np.linalg.norm(error[2:13])**2.0
    
    return J
