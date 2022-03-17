"""
Env 2D
@author: huiming zhou
"""
import numpy as np
class Env:
    def __init__(self,x_range,y_range,people,r,ang,no_env):
        self.r = r
        self.ang = ang
        self.people = people
        self.scale = 20
        self.x_range = x_range  # size of background
        self.y_range = y_range
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        self.no_env = no_env

        self.obs = self.obs_map()
    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        obs = set()
        l,r,t,d = self.x_range[0]*self.scale,self.x_range[1]*self.scale,self.y_range[1]*self.scale,self.y_range[0]*self.scale
        for i in range(l,r+1):
            obs.add((i, d))
        for i in range(l,r+1):
            obs.add((i, t))

        for i in range(d,t+1):
            obs.add((l, i))
        for i in range(d,t+1):
            obs.add((r, i))
        r = self.r
        ang = self.ang / 180 * np.pi

        for person in self.people:
            if person.robot != 1:
                if self.no_env:
                    obs.add((int(person.pos[0]*self.scale),int(person.pos[1]*self.scale)))
                else:
                    vel = person.vel
                    vel_ang = np.arctan2(vel[1], vel[0])
                    # if vel_ang < 0:
                    #     vel_ang = vel_ang + 2* np.pi
                    #print(vel,vel_ang)
                    for i in range(-r,r):
                        for j in range(-r,r):
                            d_ang = np.arctan2(j, i)
                            # if d_ang < 0 :
                            #     d_ang = d_ang + 2 * np.pi
                            delta_ang = min(abs(vel_ang - d_ang),2* np.pi - abs(vel_ang - d_ang))
                            if ((i*i + j*j) < r*r and delta_ang <= ang) or (i*i + j*j) < 5:
                                obs.add((int(person.pos[0]*self.scale+i),int(person.pos[1]*self.scale+j)))



        return obs

# class Env:
#     def __init__(self):
#         self.x_range = 51  # size of background
#         self.y_range = 31
#         self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
#                         (1, 0), (1, -1), (0, -1), (-1, -1)]
#         self.obs = self.obs_map()

#     def update_obs(self, obs):
#         self.obs = obs

#     def obs_map(self):
#         """
#         Initialize obstacles' positions
#         :return: map of obstacles
#         """

#         x = self.x_range
#         y = self.y_range
#         obs = set()

#         for i in range(x):
#             obs.add((i, 0))
#         for i in range(x):
#             obs.add((i, y - 1))

#         for i in range(y):
#             obs.add((0, i))
#         for i in range(y):
#             obs.add((x - 1, i))

#         for i in range(10, 21):
#             obs.add((i, 15))
#         for i in range(15):
#             obs.add((20, i))

#         for i in range(15, 30):
#             obs.add((30, i))
#         for i in range(16):
#             obs.add((40, i))

#         return obs
