import numpy as np

class DubinsParameters:
    '''
    Class that contains parameters for a Dubin's car path
    '''
    def __init__(self):
        self.p_s = np.zeros((3,1))
        self.chi_s = 0.0
        self.p_e = np.zeros((3,1))
        self.chi_e = 0.0
        self.radius = 0.0
        self.center_s = np.zeros((3,1))
        self.dir_s = 1
        self.center_e = np.zeros((3,1))
        self.dir_e = 1
        self.length = 0.0
        self.r1 = np.zeros((3,1))
        self.n1 = np.zeros((3,1))
        self.r2 = np.zeros((3,1))
        self.r3 = np.zeros((3,1))
        self.n3 = np.zeros((3,1))

    def update(self, 
               ps: np.ndarray, # (3x1) 
               chis: float, 
               pe: np.ndarray, # (3x1)
               chie: float, 
               R: float):
         self.p_s = ps
         self.chi_s = chis
         self.p_e = pe
         self.chi_e = chie
         self.radius = R
         self.compute_parameters()

    def compute_parameters(self):
        ps = self.p_s
        pe = self.p_e
        chis = self.chi_s
        chie = self.chi_e
        R = self.radius
        ell = np.linalg.norm(pe[0:2] - ps[0:2])

        if ell < 2 * R:
            print('Error in Dubins Parameters: Distance between nodes must be larger than 2R.')
        else:
            # Compute centers
            crs = ps + R * rotz(np.pi/2) @ np.array([[np.cos(chis)], [np.sin(chis)], [0]])
            cls = ps + R * rotz(-np.pi/2) @ np.array([[np.cos(chis)], [np.sin(chis)], [0]])
            cre = pe + R * rotz(np.pi/2) @ np.array([[np.cos(chie)], [np.sin(chie)], [0]])
            cle = pe + R * rotz(-np.pi/2) @ np.array([[np.cos(chie)], [np.sin(chie)], [0]])

            # Compute path lengths
            # Case 1: RSR
            theta = np.arctan2(cre[1,0] - crs[1,0], cre[0,0] - crs[0,0])
            L1 = np.linalg.norm(crs[0:2] - cre[0:2]) + R * (mod(theta - chis) + mod(chie - theta))

            # Case 2: RSL
            ell = np.linalg.norm(cre[0:2] - crs[0:2])
            temp = 2*R/ell
            if temp > 1.0:
                temp = 1.0
            theta = np.arctan2(cre[1,0] - crs[1,0], cre[0,0] - crs[0,0])
            L2 = np.sqrt(ell**2 - 4*R**2) + R*(mod(theta - chis) + mod(theta - np.pi - chie))

            # Case 3: LSR
            ell = np.linalg.norm(cle[0:2] - cls[0:2])
            temp = 2*R/ell
            if temp > 1.0:
                temp = 1.0
            theta = np.arctan2(cle[1,0] - cls[1,0], cle[0,0] - cls[0,0])
            L3 = np.sqrt(ell**2 - 4*R**2) + R*(mod(chis - theta) + mod(chie - theta + np.pi))

            # Case 4: LSL
            theta = np.arctan2(cle[1,0] - cls[1,0], cle[0,0] - cls[0,0])
            L4 = np.linalg.norm(cls[0:2] - cle[0:2]) + R * (mod(chis - theta) + mod(theta - chie))

            # Pick shortest path
            Ls = [L1, L2, L3, L4]
            L = min(Ls)
            idx = np.argmin(Ls)

            # Set parameters according to best path
            if idx == 0:
                # RSR
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cre
                self.dir_e = 1
                q1 = (cre - crs) / np.linalg.norm(cre - crs)
                self.r1 = crs + R * rotz(-np.pi/2) @ q1
                self.n1 = q1
                self.r2 = self.r1 + (np.linalg.norm(cre-crs) - 2*R) * q1
                self.r3 = cre + R * rotz(-np.pi/2) @ q1
                self.n3 = q1

            elif idx == 1:
                # RSL
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cle
                self.dir_e = -1
                q1 = (cle - crs) / np.linalg.norm(cle - crs)
                ell = np.linalg.norm(cle[0:2] - crs[0:2])
                theta = np.arctan2(cle[1,0] - crs[1,0], cle[0,0] - crs[0,0])
                v = np.sqrt(ell**2 - 4*R**2)
                angle = np.arccos(2*R/ell)
                self.r1 = crs + R * rotz(theta + angle - np.pi/2) @ np.array([[1],[0],[0]])
                self.n1 = rotz(theta + angle) @ np.array([[1],[0],[0]])
                self.r2 = cle + R * rotz(theta + angle + np.pi/2) @ np.array([[1],[0],[0]])
                self.r3 = pe
                self.n3 = rotz(chie) @ np.array([[1],[0],[0]])

            elif idx == 2:
                # LSR
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cre
                self.dir_e = 1
                q1 = (cre - cls) / np.linalg.norm(cre - cls)
                ell = np.linalg.norm(cre[0:2] - cls[0:2])
                theta = np.arctan2(cre[1,0] - cls[1,0], cre[0,0] - cls[0,0])
                v = np.sqrt(ell**2 - 4*R**2)
                angle = np.arccos(2*R/ell)
                self.r1 = cls + R * rotz(theta - angle + np.pi/2) @ np.array([[1],[0],[0]])
                self.n1 = rotz(theta - angle) @ np.array([[1],[0],[0]])
                self.r2 = cre + R * rotz(theta - angle - np.pi/2) @ np.array([[1],[0],[0]])
                self.r3 = pe
                self.n3 = rotz(chie) @ np.array([[1],[0],[0]])

            elif idx == 3:
                # LSL
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cle
                self.dir_e = -1
                q1 = (cle - cls) / np.linalg.norm(cle - cls)
                self.r1 = cls + R * rotz(np.pi/2) @ q1
                self.n1 = q1
                self.r2 = self.r1 + (np.linalg.norm(cle-cls) - 2*R) * q1
                self.r3 = cle + R * rotz(np.pi/2) @ q1
                self.n3 = q1

            self.length = L

    def compute_points(self):
        Del = 0.1  # distance between point

        # points along start circle
        th1 = np.arctan2(self.p_s.item(1) - self.center_s.item(1),
                         self.p_s.item(0) - self.center_s.item(0))
        th1 = mod(th1)
        th2 = np.arctan2(self.r1.item(1) - self.center_s.item(1),
                         self.r1.item(0) - self.center_s.item(0))
        th2 = mod(th2)
        th = th1
        theta_list = [th]
        if self.dir_s > 0:
            if th1 >= th2:
                while th < th2 + 2*np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2*np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)

        points = np.array([[self.center_s.item(0) + self.radius * np.cos(theta_list[0]),
                            self.center_s.item(1) + self.radius * np.sin(theta_list[0]),
                            self.center_s.item(2)]])
        for angle in theta_list:
            new_point = np.array([[self.center_s.item(0) + self.radius * np.cos(angle),
                                   self.center_s.item(1) + self.radius * np.sin(angle),
                                   self.center_s.item(2)]])
            points = np.concatenate((points, new_point), axis=0)

        # points along straight line
        sig = 0
        while sig <= 1:
            new_point = np.array([[(1 - sig) * self.r1.item(0) + sig * self.r2.item(0),
                                   (1 - sig) * self.r1.item(1) + sig * self.r2.item(1),
                                   (1 - sig) * self.r1.item(2) + sig * self.r2.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            sig += Del

        # points along end circle
        th2 = np.arctan2(self.p_e.item(1) - self.center_e.item(1),
                         self.p_e.item(0) - self.center_e.item(0))
        th2 = mod(th2)
        th1 = np.arctan2(self.r2.item(1) - self.center_e.item(1),
                         self.r2.item(0) - self.center_e.item(0))
        th1 = mod(th1)
        th = th1
        theta_list = [th]
        if self.dir_e > 0:
            if th1 >= th2:
                while th < th2 + 2*np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2*np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)
        for angle in theta_list:
            new_point = np.array([[self.center_e.item(0) + self.radius * np.cos(angle),
                                   self.center_e.item(1) + self.radius * np.sin(angle),
                                   self.center_e.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        return points

def rotz(theta: float):
    '''
    returns rotation matrix for right handed passive rotation about z-axis
    '''
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def mod(x: float):
    '''
    wrap x to be between 0 and 2*pi
    '''
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x

