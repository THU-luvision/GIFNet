# config of simulator
import numpy as np
'''generate parameters'''
num_ped = 3
maxFrame = 400
num_group = 0
groupsize = 2


'''universal law parameters'''
neib_ang = np.pi/12 # half of decision field angle
view_ang = np.pi/3 # half of decision field angle for visualize
end_thre = 0.5
s = 10  # environment size
k = 1.5
m = 2.0
t0 = 3
dt = 0.25 # Time step
rad_grp = 0.5  # group radius
rad_sin = 0.5 # single radius
sight = 10  # Neighbor search range
maxF = 5 # Maximum force/acceleration maxF=1.3 work
gv_speed = 0.7525  # goal velocity absolute speed, approx. equal to 5 km/h
maxIttr = 5000  # Max number of iterations
FOV = 60
#r = 10
'''social force parameters'''
tau = 0.5 
out_of_view_factor = 0.0

'''canvas size'''
scale = 20
height=(0,40)
width=(0,60)
start_width = (5,15)
end_width = (45,55) 
MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed
motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]

robo_vel = 7