# rrt straight line path planner for mavsim_python
import numpy as np
from message_types.msg_waypoints import MsgWaypoints

class RRTStraightLine:
    def __init__(self):
        self.segment_length = 300  # standard length of path segments

    def update(self, start_pose, end_pose, Va, world_map, radius):
        tree = MsgWaypoints()
        waypoints = MsgWaypoints()
        waypoints_not_smoothed = MsgWaypoints()
        tree.type = 'fillet'

        # add the start pose to the tree
        tree.add(start_pose, Va, np.inf, np.inf, np.inf, np.inf)
        tree.parent = np.array([[-1]])
        tree.cost = np.array([[0.0]])
        tree.connect_to_goal = np.array([[0]])

        connected_to_goal = False
        max_iterations = 500

        for _ in range(max_iterations):
            flag = self.extend_tree(tree, end_pose, Va, world_map)
            if flag:
                connected_to_goal = True
                break

        if not connected_to_goal:
            print("[Warning] Goal not reached after maximum iterations.")

        waypoints_not_smoothed = find_minimum_path(tree, end_pose)
        waypoints = smooth_path(waypoints_not_smoothed, world_map)

        self.waypoints_not_smoothed = waypoints_not_smoothed
        self.tree = tree
        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map):
        pd = tree.ned.item(2, 0)

        # Step 1: Random point
        random_point = random_pose(world_map, pd)

        # Step 2: Find nearest node
        dists = [distance(column(tree.ned, i), random_point) for i in range(tree.num_waypoints)]
        idx = np.argmin(dists)
        nearest_node = column(tree.ned, idx)

        # Step 3: Move towards random
        direction = (random_point - nearest_node)
        direction = direction / np.linalg.norm(direction)
        move_distance = min(self.segment_length, distance(nearest_node, random_point))
        new_point = nearest_node + move_distance * direction

        # Step 4: Collision check
        if collision(nearest_node, new_point, world_map):
            return False

        # Step 5: Add new node
        tree.add(new_point, Va, np.inf, np.inf, np.inf, np.inf)
        new_idx = tree.num_waypoints - 1

        if tree.parent.shape[0] < tree.num_waypoints:
            tree.parent = np.vstack((tree.parent, np.array([[idx]])))
            tree.cost = np.vstack((tree.cost, tree.cost[idx] + distance(nearest_node, new_point)))
            tree.connect_to_goal = np.vstack((tree.connect_to_goal, np.array([[0]])))
        else:
            tree.parent[new_idx] = idx
            tree.cost[new_idx] = tree.cost[idx] + distance(nearest_node, new_point)

        # Step 6: Check connection to goal
        if not collision(new_point, end_pose, world_map):
            tree.connect_to_goal[new_idx] = 1
            print(f"[RRT] Found connection to goal at node {new_idx}")
            return True

        return False


    def process_app(self):
        self.planner_viewer.process_app()


def smooth_path(waypoints, world_map):
    smooth = [0]
    for i in range(2, waypoints.num_waypoints):
        if collision(waypoints.ned[:, smooth[-1]].reshape(3,1), waypoints.ned[:, i].reshape(3,1), world_map):
            smooth.append(i-1)
    smooth.append(waypoints.num_waypoints-1)

    smooth_waypoints = MsgWaypoints()
    for idx in smooth:
        smooth_waypoints.add(column(waypoints.ned, idx),
                             waypoints.airspeed.item(idx),
                             np.inf, np.inf, np.inf, np.inf)
    smooth_waypoints.type = waypoints.type
    return smooth_waypoints


def find_minimum_path(tree, end_pose):
    # find the lowest cost path to the end node
    # find nodes that connect to end_node
    connecting_nodes = []
    for i in range(tree.num_waypoints):
        if tree.connect_to_goal.item(i) == 1:
            connecting_nodes.append(i)
    # find minimum cost last node
    idx = np.argmin(tree.cost[connecting_nodes])
    # construct lowest cost path order
    path = [connecting_nodes[idx]]  # last node that connects to end node
    parent_node = tree.parent.item(connecting_nodes[idx])
    while parent_node >= 1 and parent_node != np.inf:
        path.insert(0, int(parent_node))
        parent_node = tree.parent.item(int(parent_node))
    path.insert(0, 0)
    # construct waypoint path
    waypoints = MsgWaypoints()
    for i in path:
        waypoints.add(column(tree.ned, i),
                      tree.airspeed.item(i),
                      np.inf,
                      np.inf,
                      np.inf,
                      np.inf)
    waypoints.add(end_pose,
                  tree.airspeed[-1],
                  np.inf,
                  np.inf,
                  np.inf,
                  np.inf)
    waypoints.type = tree.type
    return waypoints


def random_pose(world_map, pd):
    # generate a random pose
    pn = world_map.city_width * np.random.rand()
    pe = world_map.city_width * np.random.rand()
    pose = np.array([[pn], [pe], [pd]])
    return pose


def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = np.linalg.norm(start_pose - end_pose)
    return d


def collision(start_pose, end_pose, world_map):
    # check to see of path from start_pose to end_pose colliding with map
    collision_flag = False
    points = points_along_path(start_pose, end_pose, 100)
    for i in range(points.shape[1]):
        if height_above_ground(world_map, column(points, i)) <= 0:
            collision_flag = True
    return collision_flag


def height_above_ground(world_map, point):
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

def points_along_path(start_pose, end_pose, N):
    # returns points along path separated by Del
    points = start_pose
    q = (end_pose - start_pose)
    L = np.linalg.norm(q)
    q = q / L
    w = start_pose
    for i in range(1, N):
        w = w + (L / N) * q
        points = np.append(points, w, axis=1)
    return points


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col