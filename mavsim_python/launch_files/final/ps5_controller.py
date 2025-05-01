# ps5_controller.py

import pygame
import numpy as np

# Initial control values
airspeed_command = 25.0
altitude_command_val = 60.0
course_command_val = np.radians(90)

# D-Pad button mappings (may vary by controller)
DPAD_UP = 11
DPAD_DOWN = 12
DPAD_LEFT = 13
DPAD_RIGHT = 14

def init_joystick():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def wrap_angle(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def get_joystick_commands(joystick):
    global airspeed_command, altitude_command_val, course_command_val

    pygame.event.pump()

    # D-Pad control for heading and altitude
    if joystick.get_button(DPAD_LEFT):
        course_command_val -= np.radians(2.0)
    if joystick.get_button(DPAD_RIGHT):
        course_command_val += np.radians(2.0)
    course_command_val = wrap_angle(course_command_val)

    if joystick.get_button(DPAD_UP):
        altitude_command_val += 1.0
    if joystick.get_button(DPAD_DOWN):
        altitude_command_val -= 1.0

    # Triggers for airspeed control
    l2 = joystick.get_axis(4)
    r2 = joystick.get_axis(5)
    l2_val = (1 - l2) / 2
    r2_val = (1 - r2) / 2

    airspeed_command += (r2_val - l2_val) * 0.5

    # Clamp
    airspeed_command = max(10, min(35, airspeed_command))
    altitude_command_val = max(0, min(200, altitude_command_val))

    return airspeed_command, altitude_command_val, course_command_val
