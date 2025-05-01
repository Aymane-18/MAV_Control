import os, sys
from pathlib import Path
import numpy as np
import time
import pygame
import termios
import tty
import atexit

# insert parent directory at beginning of python search path
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot import Autopilot
from viewers.view_manager import ViewManager
from message_types.msg_autopilot import MsgAutopilot

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation,
                      gust_flag=False,
                      steady_state=np.array([[0.0], [5.0], [0.0]]))

mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
viewers = ViewManager(animation=True, data=False, video=False, video_name='chap6.mp4')

# autopilot commands
commands = MsgAutopilot()

# =====================
# Terminal Echo Management
# =====================
def set_raw_terminal():
    global old_settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def reset_terminal():
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

atexit.register(reset_terminal)
set_raw_terminal()

# =====================
# PS5 Controller Setup
# =====================
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

airspeed_command = 25.0
altitude_command_val = 60.0
course_command_val = np.radians(90)

print("Use D-Pad (arrows) to change heading/altitude, R2/L2 to increase/decrease speed. ESC to quit.")

def wrap_angle(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def get_joystick_commands():
    global airspeed_command, altitude_command_val, course_command_val

    pygame.event.pump()

    # --- D-Pad emulated using buttons ---
    DPAD_UP = 11
    DPAD_DOWN = 12
    DPAD_LEFT = 13
    DPAD_RIGHT = 14

    if joystick.get_button(DPAD_LEFT):
        course_command_val -= np.radians(0.5)
    if joystick.get_button(DPAD_RIGHT):
        course_command_val += np.radians(0.5)
    course_command_val = course_command_val

    if joystick.get_button(DPAD_UP):
        altitude_command_val += 0.5
    if joystick.get_button(DPAD_DOWN):
        altitude_command_val -= 0.5

    # --- Triggers for speed control ---
    l2 = joystick.get_axis(4)
    r2 = joystick.get_axis(5)

    l2_val = (1 - l2) / 2
    r2_val = (1 - r2) / 2

    airspeed_command += (r2_val - l2_val) * 0.2

    # Clamp
    airspeed_command = max(10, min(35, airspeed_command))
    altitude_command_val = max(0, min(200, altitude_command_val))

# =====================
# Main Simulation Loop
# =====================
sim_time = SIM.start_time
end_time = 180

while sim_time < end_time:

    # -------joystick input-------------
    get_joystick_commands()

    # -------autopilot commands-------------
    commands.airspeed_command = airspeed_command
    commands.course_command = course_command_val
    commands.altitude_command = altitude_command_val

    # -------autopilot-------------
    estimated_state = mav.true_state  # uses true states in the control
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # ------- update viewers -------
    viewers.update(
        sim_time,
        true_state=mav.true_state,  # true states
        commanded_state=commanded_state,  # commanded states
        delta=delta,  # inputs to MAV
    )

    # -------print status-------------
    if sim_time % 1 < SIM.ts_simulation:
        print(f"[t={sim_time:.1f}s] Va={airspeed_command:.1f} | h={altitude_command_val:.1f} | chi={np.degrees(course_command_val):.1f}°")

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    time.sleep(0.002)  # slow down the simulation for visualization

viewers.close(dataplot_name="ch6_data_plot")
