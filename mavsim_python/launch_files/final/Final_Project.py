# main_sim.py

import os, sys
from pathlib import Path
import numpy as np
import time
import pygame

# insert parent directory at beginning of python search path
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

# MAVSim imports
import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot import Autopilot
from viewers.view_manager import ViewManager
from message_types.msg_autopilot import MsgAutopilot
from message_types.msg_world_map import MsgWorldMap

# Controller module
from ps5_controller import init_joystick, get_joystick_commands

# Init joystick
joystick = init_joystick()

# Initialize simulation elements
wind = WindSimulation(SIM.ts_simulation,
                      gust_flag=False,
                      steady_state=np.array([[0.0], [5.0], [0.0]]))

mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)

viewers = ViewManager(animation=True, data=True, video=False, map=True, video_name='chap6.mp4')
world_map = MsgWorldMap()
commands = MsgAutopilot()

# =====================
# Main Simulation Loop
# =====================
sim_time = SIM.start_time
end_time = 180

while sim_time < end_time:

    # --- Controller input ---
    airspeed_command, altitude_command_val, course_command_val = get_joystick_commands(joystick)

    # --- Autopilot commands ---
    commands.airspeed_command = airspeed_command
    commands.course_command = course_command_val
    commands.altitude_command = altitude_command_val

    # --- Autopilot and dynamics update ---
    estimated_state = mav.true_state
    delta, commanded_state = autopilot.update(commands, estimated_state)
    current_wind = wind.update()
    mav.update(delta, current_wind)

    # --- Viewer update ---
    viewers.update(
        sim_time,
        true_state=mav.true_state,
        estimated_state=None,
        commanded_state=commanded_state,
        delta=delta,
        path=None,
        waypoints=None,
        map=world_map
    )

    # --- Status printout ---
    if sim_time % 1 < SIM.ts_simulation:
        print(f"[t={sim_time:.1f}s] Va={airspeed_command:.1f} | h={altitude_command_val:.1f} | chi={np.degrees(course_command_val):.1f}°")

    # --- Time advance ---
    sim_time += SIM.ts_simulation
    time.sleep(0.0005)

viewers.close(dataplot_name="ch6_data_plot")
