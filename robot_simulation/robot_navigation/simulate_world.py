'''
@Time      :2021/05/21 21:26:23
@Author    :Goolo
@Desc      : simulate pedestrain walking pattern according to
            the universal law.
'''

from typing import Tuple
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
from universal_law import UnlSimulator, Person, Born
import sys
sys.path.append('../PathPlanning/Search_based_Planning/Search_2D')
from Astar import *

def check_pos(pos_list,pos):

    check = True
    pos = np.array(pos)
    for p in pos_list:
        p = np.array(p)
        if np.linalg.norm(p-pos) <= 1:
            check = False
    return check
def bron_people_ex(idxx,num_peds):
    people = []
    people1 = []
    people2 = []
    people_pos = []
    people_goal = []
    people_no_robo = []
    '''
    generate persons
    '''
    if idxx == 3:
        num_peds = 8
    for pid in range(num_peds):
        # generate pinitial positions
        zone = Born(idxx)

        pos, goal = zone.target_zone(pid)
        pos = np.array(pos) + 0.1*np.random.randn(2)
        goal = np.array(goal) + 0.1*np.random.randn(2)
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
            pr = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p1 = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            p2 = Person(pos=pos, goal=goal, vel=vel, rad=rad_sin,sight=5,ang=np.pi,nei_sight=5,nei_ang=np.pi)
            people_no_robo.append(pr)
        people.append(p)
        people1.append(p1)
        people2.append(p2)

    return people, people1,people2, people_no_robo

def main_no_robo( people_no_robo,idxx,max_fids,num_peds):
    if idxx ==3:
        num_peds = 8
    '''
    generate each frame without robot
    '''            
    trajs_no_robo = [[] for i in range(num_peds - 1)]
    vels_no_robo  = [[] for i in range(num_peds - 1)]
    simulator_no_robo = UnlSimulator(people_no_robo)
    for fid in range(max_fids):
        logger.info('update frame {}'.format(fid))
        doneflag = simulator_no_robo.update(dt,None,fid)
        # if each person arrive their goal
        if doneflag:
            break
        for i in range(len(people_no_robo)):

            trajs_no_robo[i].append(people_no_robo[i].pos.copy())
            vels_no_robo[i].append(people_no_robo[i].vel.copy())
    trajs_no_robo = np.array(trajs_no_robo)
    vels_no_robo = np.array(vels_no_robo) 

    pe_no_robo = path_efficiency(trajs_no_robo,people_no_robo)
    pi_no_robo,energy = path_irregularity(vels_no_robo,trajs_no_robo,people_no_robo)


    return pe_no_robo,pi_no_robo,energy

def main_robo(ppeople,fov,r,idxx,num_peds,dt):
    if idxx ==3:
        num_peds = 8
    '''
    generate each frame
    '''
    trajs = [[] for i in range(num_peds)]
    vels = [[] for i in range(num_peds)]
    simulator = UnlSimulator(ppeople)
    no_env = False
    for fid in range(maxFrame):
        logger.info('update frame {}'.format(fid))
        #robo setting
        s_start = (int(ppeople[0].pos[0]*scale),int(ppeople[0].pos[1]*scale))
        s_goal = (int(ppeople[0].goal[0]*scale),int(ppeople[0].goal[1]*scale))
        
        envs = env.Env(width,height,ppeople,r,fov,no_env)
        no_env = False


        if s_start in envs.obs:
            print(s_start,'s_tart in obs')
            print(dt,robo_vel,r)
            for i in range(len(ppeople)):
                print("pos at begin of for",ppeople[i].pos)
        if s_goal in envs.obs:
            print('s_goal in obs')
            print(dt,robo_vel,r)
        plot = plotting.Plotting(s_start, s_goal, envs)
        astar = AStar(s_start, s_goal, "euclidean",envs)
        try:
            path, visited = astar.searching()
        except KeyError:
            plot.plot_grid('Astar')
            print('ERROR') 
            raise
        if len(path) > robo_vel + 2:
            robo_dx = path[-1 - robo_vel][0] - path[- robo_vel][0]
            robo_dy = path[-1 - robo_vel][1] - path[- robo_vel][1]
            ang = np.arctan2(robo_dy,robo_dx)
            robo_x = int(path[-1][0] + robo_vel * np.cos(ang))
            robo_y = int(path[-1][1] + robo_vel * np.sin(ang))
            robo_step = [robo_x/scale,robo_y/scale]
        else:
            robo_step = [path[-2][0]/scale,path[-2][1]/scale]
        
        doneflag = simulator.update(dt,robo_step,fid)
        env_new = env.Env(width,height,ppeople,r,fov,no_env)
        fu_pos = (int(ppeople[0].pos[0]*scale),int(ppeople[0].pos[1]*scale))
        find = False
        if fu_pos in env_new.obs:
            robo_ang_index = motions.index((robo_dx,robo_dy))
            print(robo_ang_index)
            for i in range(1,5):
                for j in range(-1,2,2):
                    ang_new = np.arctan2(motions[(robo_ang_index + j * i)%8][1],motions[(robo_ang_index + j * i)%8][0])
                    print(ang_new)
                    robo_x_new = int(path[-1][0] + robo_vel * np.cos(ang_new))
                    robo_y_new = int(path[-1][1] + robo_vel * np.sin(ang_new))
                    if (robo_x_new,robo_y_new) not in env_new.obs:
                        find = True
                        break
                if find == True:
                    break
            if find == False:
                no_env = True
            else:
                ppeople[0].pos = np.array((robo_x_new/scale,robo_y_new/scale))
                ppeople[0].vel = np.array((robo_x_new/scale,robo_y_new/scale)) - np.array((s_start[0]/scale,s_start[1]/scale))
        plot.animation(path, visited, 'Astar', savedir, fid)
        # if each person arrive their goal
        if doneflag:
            break
        for i in range(len(ppeople)):
            if find == True:
                print("pos at end of for",ppeople[i].pos)
            
            trajs[i].append(ppeople[i].pos.copy())
            vels[i].append(ppeople[i].vel.copy())
    trajs = np.array(trajs)
    vels = np.array(vels) 
    pe = path_efficiency(trajs,ppeople)
    pi,energy = path_irregularity(vels,trajs,ppeople)
    return pe, pi, fid,energy



def path_efficiency(trajs,people):

    avarage_efficiency = np.zeros(len(trajs))
    for i in range(trajs.shape[0]):
        euclidean = people[i].path_length
        print(people[i].arrive - 1)
        for j in range(people[i].arrive - 1):
            #print("ID:{},fid:{},norm:{}".format(i,j,trajs[i][j+1]-trajs[i][j]))
            avarage_efficiency[i] += np.linalg.norm(trajs[i][j+1]-trajs[i][j])
        avarage_efficiency[i] = (euclidean - 1) / (avarage_efficiency[i]+1)                                                
    return avarage_efficiency


def path_irregularity(vels,trajs,people):

    total_amount = np.zeros(len(vels))
    energy = np.zeros(len(vels))
    for i in range(vels.shape[0]):
        for j in range(people[i].arrive):
            if people[i].robot == 1:
                if j < vels.shape[1]-1:
                    target_vel = vels[i][j+1] - vels[i][j]
                else:
                    target_vel = np.array([0,0])
                if target_vel.any() != 0:
                    total_amount[i] += 1
                    energy[i] += np.abs(np.arctan2(target_vel[1],target_vel[0]))
            else:
                target_vel = (people[i].goal - trajs[i][j]) 
                vel  = vels[i][j]
                energy[i] += np.abs(np.arctan2(target_vel[1],target_vel[0]))
                if np.abs(np.arctan2(target_vel[1],target_vel[0]) - np.arctan2(vel[1],vel[0])) > np.pi/100:
                    total_amount[i] += 1
                    
    return total_amount,energy

def json_file_write(file_path, obj):
    if not os.path.exists(os.path.dirname(os.path.abspath(file_path))):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)))
    json_string = json.dumps(obj, indent=2)
    with open(file_path, "w") as f:
        f.write(json_string)       


if __name__ == '__main__':
    for m in range(10):
        for i in [0,2,3]:
            if i == 3:
                robo_vel = 10
            people, people1,people2,people_no_robo = bron_people_ex(i,i+2)
            savename = 'unl-{}-astar-double'.format(i+2)
            save_dirs =  'data/datas' + str(m)
            savedir = os.path.join('./',save_dirs,savename)
            os.makedirs(savedir, exist_ok=True)
            pe2,pi2,fid2,eng2 = main_robo(people2,180,5,i,i+2,dt)
            pe1,pi1,fid1,eng1 = main_robo(people1,30,45,i,i+2,dt)
            pe,pi,fid,eng = main_robo(people,180,45,i,i+2,dt)
            pe_no_robo,pi_no_robo,eng_no_robo= main_no_robo(people_no_robo,i,200,i+2)
            data = {'no_robo':(pe_no_robo.tolist(),pi_no_robo.tolist(),np.mean(pe_no_robo).tolist(),np.mean(pi_no_robo).tolist(),eng_no_robo.tolist()),
                    '180-20':(pe.tolist(),pi.tolist(),np.mean(pe[1:]).tolist(),np.mean(pi[1:]).tolist(),eng.tolist()) ,
                    '180-3':  (pe2.tolist(),pi2.tolist(),np.mean(pe2[1:]).tolist(),np.mean(pi2[1:]).tolist(),eng2.tolist()), 
                    '60-20' :  (pe1.tolist(),pi1.tolist(),np.mean(pe1[1:]).tolist(),np.mean(pi1[1:]).tolist(),eng1.tolist()) }
            json_file_write(os.path.join(savedir,str(i)),data)

