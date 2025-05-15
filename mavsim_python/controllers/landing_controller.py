import numpy as np
import subprocess

class LandingController:
    def __init__(self, k_pn=1.0, R_land=160.0):
        self.k_pn = k_pn       # Proportional navigation gain
        self.R_land = R_land   # Trigger landing mode within this radius
        self.landing_mode = False
        self.popup_shown = False  # Track if popup has been shown

    def update(self, mav, target_position):
        # Relative vector (inertial)
        rel = target_position - np.array([[mav.north], [mav.east], [-mav.altitude]])
        dx, dy, dz = rel[0, 0], rel[1, 0], rel[2, 0]
        
        # Distances
        distance_2d = np.sqrt(dx**2 + dy**2)
        distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)

        if distance_2d < self.R_land:
            self.landing_mode = True

        if not self.landing_mode:
            return None  # keep following path

        # Velocity components
        V = mav.Va
        vx = V * np.cos(mav.chi)
        vy = V * np.sin(mav.chi)

        # PN guidance for lateral tracking
        Omega = (dx * vy - dy * vx) / (dx**2 + dy**2 + 1e-6)
        a_lat = self.k_pn * V * Omega
        phi_c = np.clip(np.arctan2(a_lat, 9.81), -np.radians(25), np.radians(25))

        # Vertical descent with heading correction
        gamma_c = np.clip(np.arctan2(dz, distance_2d + 1e-6), -np.radians(30), -np.radians(5))
        theta_c = gamma_c

        # Print diagnostics
        vz = -V * np.sin(gamma_c)
        t_c = mav.altitude / (vz + 1e-6)
        print(f"[Landing] dist2D={distance_2d:.1f}, dist3D={distance_3d:.1f}, tc={t_c:.2f}s")

        # Optional popup (unchanged)
        if not self.popup_shown and dz < 5.0 and t_c <= 0.5:
            self.popup_shown = True
            applescript = '''
            display dialog "Landing Done! MAV has reached the ground." ¬
            with title "Landing Complete" ¬
            with icon note ¬
            buttons {"OK"}
            '''
            subprocess.call(f"osascript -e '{applescript}'", shell=True)

        return phi_c, theta_c
