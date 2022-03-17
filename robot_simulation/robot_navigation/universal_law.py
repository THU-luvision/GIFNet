
import logging
import numpy as np
from config import *
from math import sin, cos, sqrt, exp
from logcheck import logger


class UnlSimulator():
    '''
    achieve a simulator according to universal law
    '''
    def __init__(self, people):
        self.people = people
        self.reach_goal = [False] * len(people)

    def findNeighbors(self, ang=None):
        '''
        find each person's neighbor
        '''

        for i in range(len(self.people)):
            self.people[i].neighbors = []
            self.people[i].neib_dist = []

            # 不能用实际速度来计算角度，而要用目标速度
            cur_gv = self.people[i].goal - self.people[i].pos
            vel_angle = np.arctan2(cur_gv[1], cur_gv[0])
            # vel_angle = np.arctan2(self.people[i].vel[1], self.people[i].vel[0])

            for j in range(len(self.people)):
                if i == j:
                    continue

                d = self.people[j].pos - self.people[i].pos
                d_angle = np.arctan2(d[1], d[0])
                l2 = d.dot(d)
                s2 = self.people[i].sight**2
                if ang:
                    vel_ang = np.rad2deg(vel_angle)
                    d_ang = np.rad2deg(d_angle)
                    #logger.info('i{}, l2={}, s2={}, angle={},angle={}'.format(i, l2, s2, abs(d_angle - vel_angle),abs(d_ang-vel_ang)))
                    if (l2 < s2 and  min(abs(vel_angle - d_angle),2* np.pi - abs(vel_angle - d_angle)) <= np.pi/6) or l2 <= s2/100:
                        self.people[i].neighbors.append(j)
                        self.people[i].neib_dist.append(sqrt(l2))
                else:
                    self.people[i].neighbors.append(j)
                    self.people[i].neib_dist.append(sqrt(l2))

    def dE(self, persona, personb):
        '''
        compute the interaction force between two neighbors, and return it
        '''
        # if persona.robot:
        #     pos_b = personb.pos + personb.vel * 10
        # else:
        p = personb.pos - persona.pos  # relative position
        v = persona.vel - personb.vel  # relative velocity
        dist = sqrt(p.dot(p))  # distance between neighbors

        radius = persona.rad + personb.rad
        if dist < radius:  # shrink overlapping agents
            
            radius = 0.99 * dist

        a = v.dot(v)
        b = p.dot(v)
        c = p.dot(p) - radius * radius
        discr = b * b - a * c

        if discr < 0 or -0.001 < a < 0.001:
            return np.array([0, 0])

        discr = sqrt(discr)
        t = (b - discr) / a

        if t < 0 or t > 999:
            return np.array([0, 0])

        d = k * exp(-t / t0) * (v - (v * b - p * a) / (discr)) / (a * t**m) * (m / t + 1 / t0)
        return d

    def update(self, dt,robo_pos,fid):
        '''
        update each person's state, check if everyone reach goal
        delta_t in seconds.
        '''
        robo_pos = np.array(robo_pos)
        self.findNeighbors(neib_ang)

        F = []
        for i in range(len(self.people)):
            F.append(np.zeros(2))

        for i in range(len(self.people)):
            #logger.info('p{}, neib = {}'.format(i, self.people[i].neighbors))
            
            # 1. update persons goal velocity
            person = self.people[i]
            p = person.goal - person.pos
            person.gv = p / np.sqrt(p.dot(p)) * gv_speed

            if np.linalg.norm(p) < end_thre:
                if self.reach_goal[i] is False:
                    self.people[i].arrive = fid
                self.reach_goal[i] = True
                pass
            if i == 0:
                continue
            # 2. compute force component from goal velocity
            F[i] += (self.people[i].gv - self.people[i].vel) / .5
            # logger.info('p{}, fv = {}'.format(i, F[i]))
            # F[i] += 1 * np.array([rnd.uniform(-0.2,0.2), rnd.uniform(-0.2, 0.2)])

            # 3. compute force component from neighbors
            for n, j in enumerate(self.people[i].neighbors):
                F[i] += -self.dE(self.people[i], self.people[j])
                # logger.info('p{}, fn = {}'.format(i, -dE(self.people[i], self.people[j])))

        for i in range(len(self.people)):
            if self.people[i].robot == 1:
                self.people[i].vel = np.array(robo_pos - self.people[i].pos) * 5
                self.people[i].pos = robo_pos
                #logger.info('p{}, pos={}, vel={}'.format(i, self.people[i].pos, self.people[i].vel))
                continue
            # if acceleration of person i is too large, scale it down
            f = F[i]
            mag = np.sqrt(f.dot(f))
            if mag > maxF:
                f = maxF * f / mag

            self.update_person(i, dt, f)

        done = all(self.reach_goal)
        #print(done)
        return done

    def update_person(self, i, dt, f):
        '''
        update position and velocity of one person according to original acceleration f
        '''

        self.people[i].vel += f * dt
        self.people[i].pos += self.people[i].vel * dt


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


class Born:
    '''
    generate a person with initial position and goal
    '''
    def __init__(self, setting):
        self.pos = [0.0, 0.0]
        self.goal = [0.0, 0.0]
        self.setting = setting
    def target_zone(self, idx):
        if self.setting == 0:
            if idx == 0:
                self.pos = [21, 25.05]
                self.goal = [31, 25.05]
            elif idx == 1:
                self.pos = [28, 25]
                self.goal = [15, 25]
        elif self.setting == 1:
            if idx == 0:
                self.pos = [15, 20]
                self.goal = [35, 20]
            elif idx == 1:
                self.pos = [28, 20.25]
                self.goal = [20, 20.25]
            elif idx == 2:
                self.pos = [28, 19.74]
                self.goal = [20, 19.74]
        elif self.setting == 2:
            if idx == 0:
                self.pos = [15, 20.01]
                self.goal = [30, 20]
            elif idx == 1:
                self.pos = [20.01, 15.01]
                self.goal = [20, 30]
            elif idx == 2:
                self.pos = [20.01, 25]
                self.goal = [20, 10]
            elif idx == 3:
                self.pos = [25, 20.01]
                self.goal = [10, 20]
        elif self.setting == 3:
            if idx == 0:
                self.pos = [10, 20.01]
                self.goal = [30, 20]
            elif idx == 1:
                self.pos = [20.05, 15]
                self.goal = [20, 30]
            elif idx == 2:
                self.pos = [20.01, 25.05]
                self.goal = [20, 10]
            elif idx == 3:
                self.pos = [25, 20.1]
                self.goal = [10, 20]
            elif idx == 4:
                self.pos = [20-3.53, 20.01+3.53]
                self.goal = [20+5, 20-5]
            elif idx == 5:
                self.pos = [20.01+3.53, 20+3.53]
                self.goal = [20-5, 20-5]
            elif idx == 6:
                self.pos = [20.01-3.53, 20.05-3.53]
                self.goal = [20+5, 20+5]
            elif idx == 7:
                self.pos = [20+3.53, 20-3.53]
                self.goal = [20-5, 20+5]            

        return self.pos, self.goal

    def random_zone(self):
        self.pos = [rnd.uniform(start_width[0], start_width[1]), rnd.uniform(height[0]+5, height[1]-5)]
        self.goal = [rnd.uniform(end_width[0], end_width[1]), rnd.uniform(height[0]+5, height[1]-5)]
        return self.pos, self.goal

    def circle_zone(self):
        ang = rnd.uniform(0, 2*np.pi)
        pos = [self.pos[0] + 1.5*np.cos(ang),self.pos[1] + 1.5*np.sin(ang)]
        goal = [self.goal[0] + 1.5*np.cos(ang),self.goal[1] + 1.5*np.sin(ang)]

        return pos, goal
