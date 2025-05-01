import numpy as np
from message_types.msg_waypoints import MsgWaypoints
from planners.dubins_parameters import DubinsParameters

class RRTDubins:
    def __init__(self):
        self.segment_length = 450  # will be overwritten in update
        self.dubins_path = DubinsParameters()

    def update(self, start_pose, end_pose, Va, world_map, radius):
        self.segment_length = 4 * radius
        pd = start_pose[2, 0]

        tree = MsgWaypoints()
        tree.type = 'dubins'
        tree.add(start_pose[0:3], Va, start_pose[3, 0], np.inf, np.inf, np.inf)
        tree.parent = np.array([[-1]])
        tree.cost = np.array([[0.0]])
        tree.connect_to_goal = np.array([[0]])

        max_iterations = 500  # make it smaller for easier debugging
        goal_reached = False

        for _ in range(max_iterations):
            success = self.extendTree(tree, end_pose, Va, world_map, radius)
            if success:
                goal_reached = True
                break

        if not goal_reached:
            print("[RRT-Dubins] ⚠️ Goal not reached after maximum iterations.")
            self.tree = tree

        waypoints_not_smooth = findMinimumPath(tree, end_pose)

        if waypoints_not_smooth.num_waypoints == 0:
            print("[RRTDubins] ❌ No nodes can connect to goal.")
            self.waypoints_not_smooth = MsgWaypoints()
            return MsgWaypoints()  # empty safe return

        waypoints = self.smoothPath(waypoints_not_smooth, world_map, radius)
        self.tree = tree
        self.waypoints_not_smooth = waypoints_not_smooth  # ← correct spelling
        return waypoints



    def extendTree(self, tree, end_pose, Va, world_map, radius):
        pd = tree.ned[2, 0]
        random_pose = randomPose(world_map, pd, end_pose)

        dists = [distance(column(tree.ned, i), random_pose[0:3]) for i in range(tree.num_waypoints)]
        idx = np.argmin(dists)
        nearest_node = column(tree.ned, idx)
        nearest_course = tree.course.item(idx)
        start = np.vstack((nearest_node, nearest_course))
        
        dubins_path = DubinsParameters()
        try:
            dubins_path.update(start, random_pose, radius)
        except:
            return False

        if dubins_path.length > self.segment_length:
            print("[RRT-Dubins] Segment too long, skipping.")
            return False

        if self.collision(start, random_pose, world_map, radius):
            return False

        tree.add(random_pose[0:3], Va, random_pose[3, 0], np.inf, np.inf, np.inf)
        new_idx = tree.num_waypoints - 1
        tree.parent = np.vstack((tree.parent, np.array([[idx]])))
        tree.cost = np.vstack((tree.cost, tree.cost[idx] + dubins_path.length))
        tree.connect_to_goal = np.vstack((tree.connect_to_goal, np.array([[0]])))

        try:
            goal_path = DubinsParameters()
            goal_path.update(random_pose, end_pose, radius)
            if goal_path.length < self.segment_length and not self.collision(random_pose, end_pose, world_map, radius):
                tree.connect_to_goal[new_idx] = 1
                return True
        except:
            pass

        return False



    def collision(self, start_pose, end_pose, world_map, radius):
        try:
            dubins_path = DubinsParameters()
            dubins_path.update(start_pose, end_pose, radius)
        except:
            print("[RRT-Dubins] Failed to compute Dubins path.")
            return True

        for i in range(dubins_path.n_points):
            point = dubins_path.path[:, [i]]
            if heightAboveGround(world_map, point) <= 0:
                return True
        return False


    def smoothPath(self, waypoints, world_map, radius):
        smooth = [0]
        for i in range(2, waypoints.num_waypoints):
            if self.collision(waypoints.pose[:, smooth[-1]].reshape(4,1), waypoints.pose[:, i].reshape(4,1), world_map, radius):
                smooth.append(i-1)
        smooth.append(waypoints.num_waypoints-1)

        smooth_waypoints = MsgWaypoints()
        for idx in smooth:
            smooth_waypoints.add(column(waypoints.ned, idx),
                                waypoints.airspeed.item(idx),
                                waypoints.course.item(idx),
                                np.inf, np.inf, np.inf)
        smooth_waypoints.type = 'dubins'
        return smooth_waypoints



# Support functions
def findMinimumPath(tree, end_pose):
    connecting_nodes = []
    for i in range(tree.num_waypoints):
        if tree.connect_to_goal.item(i) == 1:
            connecting_nodes.append(i)

    if len(connecting_nodes) == 0:
        print("[RRTDubins] ❌ No nodes can connect to goal.")
        return MsgWaypoints()

    idx = np.argmin(tree.cost[connecting_nodes])
    path = [connecting_nodes[idx]]
    parent_node = tree.parent.item(connecting_nodes[idx])
    while parent_node >= 1:
        path.insert(0, int(parent_node))
        parent_node = tree.parent.item(int(parent_node))
    path.insert(0, 0)

    waypoints = MsgWaypoints()
    for i in path:
        waypoints.add(column(tree.ned, i),
                      tree.airspeed.item(i),
                      tree.course.item(i),
                      np.inf, np.inf, np.inf)

    waypoints.add(end_pose[0:3],
                  tree.airspeed[-1],
                  end_pose.item(3),
                  np.inf, np.inf, np.inf)
    waypoints.type = tree.type
    return waypoints



def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = np.linalg.norm(start_pose[0:3] - end_pose[0:3])
    return d


def heightAboveGround(world_map, point):
    # find the altitude of point above ground level
    point_height = -point.item(2)
    tmp = np.abs(point.item(0)-world_map.building_north)
    d_n = np.min(tmp)
    idx_n = np.argmin(tmp)
    tmp = np.abs(point.item(1)-world_map.building_east)
    d_e = np.min(tmp)
    idx_e = np.argmin(tmp)
    if (d_n<world_map.building_width) and (d_e<world_map.building_width):
        map_height = world_map.building_height[idx_n, idx_e]
    else:
        map_height = 0
    h_agl = point_height - map_height
    return h_agl


def randomPose(world_map, pd, end_pose=None):
    if end_pose is not None and np.random.rand() < 0.05:  # 5% chance
        # goal biasing
        pn = end_pose[0, 0] + np.random.uniform(-50, 50)
        pe = end_pose[1, 0] + np.random.uniform(-50, 50)
        chi = np.random.uniform(-np.pi, np.pi)
    else:
        # uniform random
        pn = world_map.city_width * np.random.rand()
        pe = world_map.city_width * np.random.rand()
        chi = np.random.uniform(-np.pi, np.pi)

    pose = np.array([[pn], [pe], [pd], [chi]])
    return pose



def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col