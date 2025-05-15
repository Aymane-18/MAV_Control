

import numpy as np
from planners.dubins_parameters import DubinsParameters
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_waypoints import MsgWaypoints

class PathManager:
    def __init__(self):
        self._path = MsgPath()
        self._num_waypoints = 0
        self._ptr_previous = 0
        self._ptr_current = 1
        self._ptr_next = 2
        self._halfspace_n = np.inf * np.ones((3,1))
        self._halfspace_r = np.inf * np.ones((3,1))
        self._manager_state = 1
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()
        self._orbit_counter = 0  # orbit timeout counter

    def update(self, waypoints: MsgWaypoints, state: MsgState, radius: float) -> MsgPath:
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints and waypoints.flag_waypoints_changed:
            self.manager_requests_waypoints = False
            self._num_waypoints = waypoints.num_waypoints
            self._initialize_pointers()
            self._manager_state = 1

        if waypoints.type == 'straight_line':
            self._line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self._fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            self._dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self._path

    def _line_manager(self, waypoints: MsgWaypoints, state: MsgState):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        if waypoints.flag_waypoints_changed:
            self._num_waypoints = waypoints.num_waypoints
            self._initialize_pointers()
        if self._inHalfSpace(mav_pos):
            if self._ptr_current == self._num_waypoints - 1:
                self.manager_requests_waypoints = True
            else:
                self._increment_pointers()
        self._construct_line(waypoints)

    def _fillet_manager(self, waypoints: MsgWaypoints, radius: float, state: MsgState):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T

        # Only process changed waypoints once
        if waypoints.flag_waypoints_changed:
            self._num_waypoints = waypoints.num_waypoints
            self._initialize_pointers()
            self._manager_state = 1
            waypoints.flag_waypoints_changed = False  # <- prevent repeat reset

        if self._manager_state == 1:
            self._construct_fillet_line(waypoints, radius)
            if self._inHalfSpace(mav_pos):
                print("[Transition] Switching from line to circle!")
                self._manager_state = 2
                self._orbit_counter = 0
                self._construct_fillet_circle(waypoints, radius)

        elif self._manager_state == 2:
            if self._orbit_counter == 0:
                print("[DEBUG] Constructing circle once")
                self._construct_fillet_circle(waypoints, radius)

            distance = np.dot((mav_pos - self._halfspace_r).T, self._halfspace_n).item()
            print(f"[DEBUG] Orbit exit check: distance = {distance:.2f}, counter = {self._orbit_counter}")

            if distance > -1 or self._orbit_counter > 100:
                print("[Transition] Exiting circle and moving to next segment.")
                if self._ptr_current == self._num_waypoints - 1:
                    self.manager_requests_waypoints = True
                else:
                    self._increment_pointers()
                    self._manager_state = 1
                    self._construct_fillet_line(waypoints, radius)
                    self._orbit_counter = 0
            else:
                self._orbit_counter += 1





    def _dubins_manager(self, waypoints: MsgWaypoints, radius: float, state: MsgState):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T

        if waypoints.flag_waypoints_changed:
            self._num_waypoints = waypoints.num_waypoints
            self._initialize_pointers()
            self._manager_state = 1
            ps = waypoints.ned[:, self._ptr_previous].reshape((3,1))
            chis = waypoints.course.item(self._ptr_previous)
            pe = waypoints.ned[:, self._ptr_current].reshape((3,1))
            chie = waypoints.course.item(self._ptr_current)
            self.dubins_path.update(ps, chis, pe, chie, radius)

        if self._manager_state == 1:
            self._construct_dubins_circle_start(waypoints, self.dubins_path)
            if self._inHalfSpace(mav_pos):
                print("[Transition] Exiting start orbit, moving to straight line.")
                self._manager_state = 2
                self._orbit_counter = 0  # Reset counter

        elif self._manager_state == 2:
            self._construct_dubins_line(waypoints, self.dubins_path)
            if self._inHalfSpace(mav_pos):
                print("[Transition] Exiting straight line, moving to end orbit.")
                self._manager_state = 3

        elif self._manager_state == 3:
            self._construct_dubins_circle_end(waypoints, self.dubins_path)
            if self._inHalfSpace(mav_pos) or self._orbit_counter > 300:
                print("[Transition] Exiting end orbit, moving to next waypoint!")
                if self._ptr_current == self._num_waypoints - 1:
                    self.manager_requests_waypoints = True
                else:
                    self._increment_pointers()
                    ps = waypoints.ned[:, self._ptr_previous].reshape((3,1))
                    chis = waypoints.course.item(self._ptr_previous)
                    pe = waypoints.ned[:, self._ptr_current].reshape((3,1))
                    chie = waypoints.course.item(self._ptr_current)
                    self.dubins_path.update(ps, chis, pe, chie, radius)
                    self._manager_state = 1
            else:
                self._orbit_counter += 1




    def _initialize_pointers(self):
        if self._num_waypoints >= 3:
            self._ptr_previous = 0
            self._ptr_current = 1
            self._ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def _increment_pointers(self):
        self._ptr_previous = (self._ptr_previous + 1) % self._num_waypoints
        self._ptr_current = (self._ptr_current + 1) % self._num_waypoints
        self._ptr_next = (self._ptr_next + 1) % self._num_waypoints

    def _construct_line(self, waypoints: MsgWaypoints):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        current = waypoints.ned[:, self._ptr_current:self._ptr_current+1]
        q_prev = (current - previous) / np.linalg.norm(current - previous)
        self._halfspace_n = q_prev
        self._halfspace_r = current
        self._path.flag = 1
        self._path.airspeed = waypoints.airspeed.item(self._ptr_current)
        self._path.line_origin = previous.flatten()
        self._path.line_direction = q_prev.flatten()

    def _construct_fillet_line(self, waypoints: MsgWaypoints, radius: float):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        current = waypoints.ned[:, self._ptr_current:self._ptr_current+1]
        next_wp = waypoints.ned[:, self._ptr_next:self._ptr_next+1]
        q_prev = (current - previous) / np.linalg.norm(current - previous)
        q_curr = (next_wp - current) / np.linalg.norm(next_wp - current)
        angle = np.arccos(-q_prev.T @ q_curr)
        z = current - (radius / np.tan(angle/2)) * q_prev
        self._halfspace_r = z
        self._halfspace_n = q_prev
        self._path.flag = 1
        self._path.airspeed = waypoints.airspeed.item(self._ptr_current)
        self._path.line_origin = z.flatten()
        self._path.line_direction = q_prev.flatten()

    def _construct_fillet_circle(self, waypoints: MsgWaypoints, radius: float):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        current = waypoints.ned[:, self._ptr_current:self._ptr_current+1]
        next_wp = waypoints.ned[:, self._ptr_next:self._ptr_next+1]
        q_prev = (current - previous) / np.linalg.norm(current - previous)
        q_curr = (next_wp - current) / np.linalg.norm(next_wp - current)

        # Ensure unit vector
        q_prev /= np.linalg.norm(q_prev)
        q_curr /= np.linalg.norm(q_curr)

        angle = np.arccos(-q_prev.T @ q_curr)
        c = current - (radius / np.sin(angle/2)) * (q_prev - q_curr) / np.linalg.norm(q_prev - q_curr)
        z2 = current + (radius / np.tan(angle/2)) * q_curr

        dot_check = (z2 - current).T @ q_curr
        print(f"  Dot (z2 - current) · q_curr = {dot_check.item():.3f}")

        if dot_check.item() < 0:
            print("⚠️ Reversing q_curr direction to face outward.")
            q_curr = -q_curr
            q_curr /= np.linalg.norm(q_curr)  # Renormalize after flip

        self._path.flag = 2
        self._path.airspeed = waypoints.airspeed.item(self._ptr_current)
        self._path.orbit_center = c.flatten()
        self._path.orbit_radius = radius
        self._path.orbit_direction = 1 if np.sign(q_prev[0]*q_curr[1] - q_prev[1]*q_curr[0]) > 0 else -1
        self._halfspace_r = z2
        self._halfspace_n = q_curr

        print("[Circle Constructed]")
        print(f"  Center c: {c.flatten()}")
        print(f"  Exit point z2: {z2.flatten()}")
        print(f"  Exit normal q_curr: {q_curr.flatten()}")
        print(f"  Orbit direction: {'CW' if self._path.orbit_direction == 1 else 'CCW'}")



    def _construct_dubins_circle_start(self, waypoints: MsgWaypoints, dubins_path: DubinsParameters):
        self._path.flag = 2
        self._path.airspeed = waypoints.airspeed.item(self._ptr_current)
        self._path.orbit_center = dubins_path.center_s.flatten()
        self._path.orbit_radius = dubins_path.radius
        self._path.orbit_direction = dubins_path.dir_s
        self._halfspace_r = dubins_path.r1
        self._halfspace_n = dubins_path.n1
        print("[Dubins] Start Circle")
        print(f"  Halfspace r1: {self._halfspace_r.flatten()}, n1: {self._halfspace_n.flatten()}")

    def _construct_dubins_line(self, waypoints: MsgWaypoints, dubins_path: DubinsParameters):
        self._path.flag = 1
        self._path.airspeed = waypoints.airspeed.item(self._ptr_current)
        self._path.line_origin = dubins_path.r1.flatten()
        self._path.line_direction = (dubins_path.r2.flatten() - dubins_path.r1.flatten())
        self._path.line_direction /= np.linalg.norm(self._path.line_direction)
        self._halfspace_r = dubins_path.r2
        self._halfspace_n = dubins_path.n1
        print("[Dubins] Straight Line")
        print(f"  Halfspace r2: {self._halfspace_r.flatten()}, n1: {self._halfspace_n.flatten()}")

    def _construct_dubins_circle_end(self, waypoints: MsgWaypoints, dubins_path: DubinsParameters):
        self._path.flag = 2
        self._path.airspeed = waypoints.airspeed.item(self._ptr_current)
        self._path.orbit_center = dubins_path.center_e.flatten()
        self._path.orbit_radius = dubins_path.radius
        self._path.orbit_direction = dubins_path.dir_e
        self._halfspace_r = dubins_path.r3
        self._halfspace_n = dubins_path.n3
        print("[Dubins] End Circle")
        print(f"  Halfspace r3: {self._halfspace_r.flatten()}, n3: {self._halfspace_n.flatten()}")


    def _inHalfSpace(self, pos: np.ndarray) -> bool:
        distance = (pos - self._halfspace_r).T @ self._halfspace_n
        if abs(distance.item()) < 20.0:
            print(f"[Halfspace Close] Distance = {distance.item():.2f}")
        return distance.item() > -1