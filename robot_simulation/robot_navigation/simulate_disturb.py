import numpy as np
import random as rnd
import os.path as osp
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib
import json
from logcheck import logger
from config import *
from universal_law import UnlSimulator
import sys
sys.path.append('../PathPlanning/Search_based_Planning/Search_2D')
from Astar import *
class Person:
    '''
    class definition for a single person having position pos, velocity vel, goal velocity gv, goal position goal,
    radius rad, neighbors neighbors, time-to-collision with neighbors nt
    '''
    def __init__(self, pos=[0.0, 0.0], vel=[0.01, 0.01], gv=[0.01, 0.01], goal=[0.0, 0.0], rad=0.5, sight=5, ang=np.pi,nei_sight=5,nei_ang=np.pi):

        self.pos = np.array(pos, dtype=float)
        self.gv = np.array(gv, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.rad = rad
        self.neighbors = []
        self.neib_dist = []
        self.sight = sight
        self.ang = ang
        self.nei_sight = nei_sight
        self.nei_ang = nei_ang
        self.robot = 0
        self.path_length = np.linalg.norm(self.pos-self.goal)
        self.target = self.goal-self.pos
        self.arrive = 0



def check_pos(pos_list,pos):
    check = True
    pos = np.array(pos)
    for p in pos_list:
        p = np.array(p)
        if np.linalg.norm(p-pos) <= 1:
            check = False
    return check

def born_people_group(idxx):
    people = []
    people1 = []
    people2 = []
    distance = distances[idxx]
    '''
    generate persons
    '''
    center_pos = [28,20]
    center_goal = [20,20]
    ang = rnd.uniform(0, 2*np.pi)
    #print(distance)
    for pid in range(num_ped):
        # generate pinitial positions
        if pid == 0:
            pos = [15, 20]
            goal = [35, 20]
        elif pid == 1:
            pos = [center_pos[0] + distance*np.cos(ang)/100.0,center_pos[1] + distance*np.sin(ang)/100.0]
            goal = [center_goal[0] + distance*np.cos(ang)/100.0,center_goal[1] + distance*np.sin(ang)/100.0]
        elif pid == 2:
            pos = [center_pos[0] - distance*np.cos(ang)/scale,center_pos[1] - distance*np.sin(ang)/scale]
            goal = [center_goal[0] - distance*np.cos(ang)/scale,center_goal[1] - distance*np.sin(ang)/scale]          
    # get velocity unit vector
        vel = np.array(goal) - np.array(pos)
        vel_norm = np.linalg.norm(vel)
        vel = vel / vel_norm
        if pid == 0:
            p = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p1 = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p2 = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p.robot = 1
            p1.robot = 1
            p2.robot = 1
        else:
            p = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p1 = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p2 = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
        people.append(p)
        people1.append(p1)
        people2.append(p2)
    return people,people1,people2

def main_disturb(ppeople,fov,r,idxx,num_peds):
    is_disturbs = 0
    for fid in range(1):

        s_start = (int(ppeople[0].pos[0]*scale),int(ppeople[0].pos[1]*scale))
        s_goal = (int(ppeople[0].goal[0]*scale),int(ppeople[0].goal[1]*scale))
        
        envs = env.Env(width,height,ppeople,r,fov,False)
        if s_start in envs.obs:
            print(s_start,'s_tart in obs')

        if s_goal in envs.obs:
            print('s_goal in obs')

        astar = AStar(s_start, s_goal, "euclidean",envs)
        try:
            path, visited = astar.searching()
        except KeyError:
            plot.plot_grid('Astar')
            print('ERROR')
            raise
        
        for r_pos in path[::-1]:
            vector_1 = ppeople[1].pos*scale - r_pos
            vector_2 = ppeople[2].pos*scale - r_pos
            ang1 = np.arctan2(vector_1[1],vector_1[0])
            ang2 = np.arctan2(vector_2[1],vector_2[0])
            delta_ang = min(abs(ang1 - ang2),2* np.pi - abs(ang1 - ang2))
            if delta_ang > np.pi * 5 / 6:
                is_disturbs = 1
                return is_disturbs

    return is_disturbs


def json_file_write(file_path, obj):
    if not os.path.exists(os.path.dirname(os.path.abspath(file_path))):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)))
    json_string = json.dumps(obj, indent=2)
    with open(file_path, "w") as f:
        f.write(json_string)       

if __name__ == '__main__':
    distances = [10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,65]
    resutls = []
    for idx in range(len(distances)):
        is_disturb = 0
        is_disturb1 = 0
        is_disturb2 = 0
        for i in range(200):
            people,people1,people2 = born_people_group(idx)
            is_disturb += main_disturb(people, 180, 5, idx, 3)
            is_disturb1 += main_disturb(people1, 30, 45, idx, 3)
            is_disturb2 += main_disturb(people2, 180, 45, idx, 3)
        resutls.append([is_disturb/200.0,is_disturb1/200.0,is_disturb2/200.0])
    print('result')
    print(resutls)
