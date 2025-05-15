import os, sys
from pathlib import Path
import numpy as np
import time
from pynput import keyboard

# insert parent directory at beginning of python search path
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot_lqr import Autopilot
from viewers.view_manager import ViewManager
from message_types.msg_autopilot import MsgAutopilot
from message_types.msg_world_map import MsgWorldMap


wind = WindSimulation(SIM.ts_simulation,
                      gust_flag=False,
                      steady_state=np.array([[0.0], [5.0], [0.0]]))

mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)

# Enable map rendering
viewers = ViewManager(animation=True, data=False, video=False, map=True, video_name='chap6.mp4')
world_map = MsgWorldMap()

# Autopilot commands
commands = MsgAutopilot()

# =====================
# Keyboard Control Setup
# =====================
airspeed_command = 30.0
altitude_command_val = 0.0
course_command_val = np.radians(0)

print("Use W/S to climb/descend, A/D to turn, Q/E to slow/speed up. Press ESC to quit.")

def on_press(key):
    global airspeed_command, altitude_command_val, course_command_val
    try:
        if key.char == 'w':
            altitude_command_val += 1
        elif key.char == 's':
            altitude_command_val -= 1
        elif key.char == 'a':
            course_command_val -= np.radians(10)
        elif key.char == 'd':
            course_command_val += np.radians(10)
        elif key.char == 'q':
            airspeed_command = max(5, airspeed_command - 1)
        elif key.char == 'e':
            airspeed_command = min(35, airspeed_command + 1)
    except AttributeError:
        if key == keyboard.Key.esc:
            print("Exiting simulation.")
            os._exit(0)

listener = keyboard.Listener(on_press=on_press)
listener.start()

# =====================
# Main Simulation Loop
# =====================
sim_time = SIM.start_time
end_time = 180

while sim_time < end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = airspeed_command
    commands.course_command = course_command_val
    commands.altitude_command = altitude_command_val

    # -------autopilot-------------
    estimated_state = mav.true_state
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()
    mav.update(delta, current_wind)

    # ------- update viewers -------
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


    # -------print status-------------
    if sim_time % 1 < SIM.ts_simulation:
        print(f"[t={sim_time:.1f}s] Va={airspeed_command:.1f} | h={altitude_command_val:.1f} | chi={np.degrees(course_command_val):.1f}°")

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    #time.sleep(0.000001)

viewers.close(dataplot_name="ch6_data_plot")
