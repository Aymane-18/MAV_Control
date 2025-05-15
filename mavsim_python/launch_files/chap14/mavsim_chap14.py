"""
mavsim_python - Chapter 14 visual servoing and precision landing
Beard & McLain, PUP, 2012
"""

import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import pyqtgraph as pg
import parameters.simulation_parameters as SIM
import parameters.planner_parameters as PLAN

import numpy as np
from models.wind_simulation import WindSimulation
from models.camera import Camera
from models.target_dynamics import TargetDynamics
from models.mav_dynamics_camera import MavDynamics
from models.gimbal import Gimbal
from controllers.autopilot import Autopilot
from controllers.landing_controller import LandingController
from estimators.observer import Observer
from estimators.geolocation import Geolocation
from planners.path_follower import PathFollower
from planners.path_manager_follow_target import PathManager
from message_types.msg_world_map_target import MsgWorldMap
from message_types.msg_waypoints import MsgWaypoints
from viewers.geolocation_viewer import GeolocationViewer
from viewers.data_viewer import DataViewer
from viewers.mav_world_camera_viewer import MAVWorldCameraViewer
from viewers.camera_viewer import CameraViewer

# Visualization toggles
VIDEO = False
DATA_PLOTS = False
ANIMATION = True
GEO_PLOTS = True
CAMERA_VIEW = True

if VIDEO:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap14_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# Initialize Qt app and viewers
if ANIMATION or DATA_PLOTS or GEO_PLOTS:
    app = pg.QtWidgets.QApplication([])
if ANIMATION:
    world_view = MAVWorldCameraViewer(app=app)
if DATA_PLOTS:
    data_view = DataViewer(app=app, dt=SIM.ts_simulation,
                           plot_period=SIM.ts_plot_refresh,
                           data_recording_period=SIM.ts_plot_record_data,
                           time_window_length=30)
if GEO_PLOTS:
    geo_viewer = GeolocationViewer(app=app, dt=SIM.ts_simulation,
                                   plot_period=SIM.ts_plot_refresh,
                                   data_recording_period=SIM.ts_plot_record_data,
                                   time_window_length=30)
if CAMERA_VIEW:
    camera_view = CameraViewer()

# Initialize simulation elements
world_map = MsgWorldMap()
mav = MavDynamics(SIM.ts_simulation)
gimbal = Gimbal()
camera = Camera()
target = TargetDynamics(SIM.ts_simulation, world_map)
wind = WindSimulation(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
observer = Observer(SIM.ts_simulation)
path_follower = PathFollower()
path_manager = PathManager()
waypoints = MsgWaypoints()
geolocation = Geolocation(SIM.ts_simulation)
landing_controller = LandingController()

# Simulation loop
sim_time = SIM.start_time
print("Press Command-Q to exit...")

while sim_time < SIM.end_time:
    # ------- Sensors / Camera -------
    measurements = mav.sensors()
    camera.updateProjectedPoints(mav.true_state, target.position())
    pixels = camera.getPixels()
    estimated_state = mav.true_state  # or use observer.update(measurements)

    # ------- Geolocation / Target Estimation -------
    estimated_target_position = geolocation.update(estimated_state, pixels)

    # ------- Gimbal Pointing -------
    gimbal_cmd = gimbal.pointAtPosition(estimated_state, target.position())
    delta_gimbal_az = gimbal_cmd.item(0)
    delta_gimbal_el = gimbal_cmd.item(1)

    # ------- Path Management -------
    path = path_manager.update(target.position())

    # ------- Autopilot Command -------
    autopilot_commands = path_follower.update(path, estimated_state)

    # ------- Landing Controller Integration -------
    landing_cmd = landing_controller.update(estimated_state, target.position())
    if landing_cmd is not None:
        phi_c, theta_c = landing_cmd
        autopilot_commands.phi_feedforward = phi_c
        autopilot_commands.altitude_command = -mav.true_state.altitude + 10 * np.tan(theta_c)


    # ------- Control / Actuation -------
    delta, commanded_state = autopilot.update(autopilot_commands, estimated_state)
    delta.gimbal_az = delta_gimbal_az
    delta.gimbal_el = delta_gimbal_el

    # ------- Dynamics Integration -------
    wind_vector = wind.update()
    mav.update(delta, wind_vector)
    target.update()

    # ------- Viewer Updates -------
    if ANIMATION:
        world_view.update(mav.true_state, target.position(), path, waypoints, world_map)
    if DATA_PLOTS:
        data_view.update(mav.true_state, estimated_state, commanded_state, delta)
    if GEO_PLOTS:
        geo_viewer.update(estimated_target_position - target.position())
    if CAMERA_VIEW:
        camera_view.updateDisplay(camera.getProjectedPoints())
    if ANIMATION or DATA_PLOTS or GEO_PLOTS:
        app.processEvents()

    sim_time += SIM.ts_simulation

if VIDEO:
    video.close()
