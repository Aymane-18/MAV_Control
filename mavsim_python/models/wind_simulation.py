"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag=True, steady_state=np.array([[0., 0., 0.]]).T):
        self._steady_state = steady_state
        self._Ts = Ts

        # Dryden model parameters for 50m, light turbulence
        Lu = Lv = 200.0
        Lw = 50.0
        sigma_u = sigma_v = 1.06
        sigma_w = 0.7
        Va = 25.0  # Nominal airspeed (same as Va_trim)

        # Transfer functions from white noise to gust (Dryden model)
        self.u_w = TransferFunction(
            num=np.array([[sigma_u * np.sqrt(2 * Va / Lu)]]),
            den=np.array([[1.0, Va / Lu]]),
            Ts=Ts
        )

        self.v_w = TransferFunction(
            num=np.array([[sigma_v * np.sqrt(3 * Va / Lv), sigma_v * np.sqrt(3 * Va / Lv) * Va / Lv]]),
            den=np.array([[1.0, 2 * Va / Lv, (Va / Lv)**2]]),
            Ts=Ts
        )

        self.w_w = TransferFunction(
            num=np.array([[sigma_w * np.sqrt(3 * Va / Lw), sigma_w * np.sqrt(3 * Va / Lw) * Va / Lw]]),
            den=np.array([[1.0, 2 * Va / Lw, (Va / Lw)**2]]),
            Ts=Ts
        )

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))
