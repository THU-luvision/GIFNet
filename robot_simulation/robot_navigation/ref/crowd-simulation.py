import random as rnd
from tkinter import *
import time
from math import sin, cos, sqrt, exp
import numpy as np

# various environmental and agent parameters
num = 0         # initial number of agents
s = 10          # environment size
k = 1.5
m = 2.0
t0 = 3
dt = 0.05       # Time step
rad = 0.22      # Collision radius
sight = 2.2     # Neighbor search range
maxF = 5        # Maximum force/acceleration
gv_speed = 0.7525  # goal velocity absolute speed, approx. equal to 5 km/h
maxIttr = 5000  # Max number of iterations

pixelsize = 600
scale = pixelsize / s

h = 30*15           # height
height = h / scale  # window height
w = 116*15          # width
width = w / scale   # window width

framedelay = 30
drawVels = True
win = Tk()
canvas = Canvas(win, width=w, height=h, background="#444")

# create list of walls
walllist = [(0, 22*15, 25*15-15, 22*15), (28*15+15, 22*15, 48*15, 22*15), (48*15, 22*15, 48*15, 25*15),
            (48*15, 25*15, 68*15, 25*15), (68*15, 25*15, 68*15, 22*15),
            (68*15, 22*15, 76.5*15-15, 22*15), (79.5*15+15, 22*15, 113*15-15, 22*15), # exit 0
            (3*15+15, 8*15, 36.5*15-15, 8*15), (39.5*15+15, 8*15, 48*15, 8*15), # exit 1
            (48*15, 8*15, 48*15, 5*15), (48*15, 5*15, 68*15, 5*15), (68*15, 5*15, 68*15, 8*15),
            (68*15, 8*15, 88*15-15, 8*15), (91*15+15, 8*15, 116*15, 8*15)]

for wall in walllist:
    line = canvas.create_line(wall)  # draw walls on the canvas

walllist = np.array(walllist)/scale  # covert wall list to np.array


class Person:
    # class definition for a single person having position pos, velocity vel, goal velocity gv, goal position goal,
    # radius rad, neighbors neighbors, time-to-collision with neighbors nt
    def __init__(self, pos=[0.0, 0.0], vel=[0.01, 0.01], gv=[0.01, 0.01], goal=[0.0, 0.0], coffee=0, watch=False):
        self.vel = np.array(vel)
        self.pos = np.array(pos)
        self.gv = np.array(gv)
        self.goal = np.array(goal)
        self.goal_save = np.array(goal)
        self.rad = rad
        self.coffeeCharge = coffee  # does the pgiven person crave coffee?
        self.inline = False  # Is the person in a coffee line
        self.neighbors = []
        self.nt = []


class Emitter:
    # class definition for an emitter that emitts people in a given rectangle according to a Poisson process with
    # parameter lam
    def __init__(self, xmin=0.0, xmax=0.0, ymin=0.0, ymax=0.0, lam=0.05):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.lam = lam
        self.spawn = np.random.poisson(lam=lam, size=maxIttr)


class Zone:
    # class definiton of zones having a defined rectangular domain and a given velocity field for passer-bys
    def __init__(self, pos=[0.0, 0.0, 0.0, 0.0], field=[1.0, 0.0]):
        self.pos = np.array(pos)/scale
        self.field = np.array(field)


class coffeeStand:
    # class definition of coffee stands having a defined rectangular domain, charge = {0, 1} (1 means that the stand has
    # coffee and agents can approach the stand to get given coffee, 0 means that the stand doesn't have coffee or agents
    # cannot approach the stand to get coffee), and a given number of remaining coffee cups
    def __init__(self, pos_x=[0.0, 0.0], pos_y=[0.0, 0.0], charge=1, coffeeCupsLeft=10, imgpos=[0.0, 0.0]):
        self.pos_x = np.array(pos_x)
        self.pos_y = np.array(pos_y)
        self.pos = np.array([np.mean(self.pos_x), np.mean(self.pos_y)])  # center of the coffee stand
        self.imgpos = imgpos  # used for plotting coffee count on canvas
        self.id = None  # used for updating coffee count on canvas
        self.charge = charge  # is the stand active or not?
        self.coffeeCupsLeft = coffeeCupsLeft  # how many coffee cups are left?
        self.inline = 0  # how many persons are currently in line for coffee

    def studentCoffee(self):
        # define a member function for each stand: if a person takes a cup of coffee from the stand, reduce the number of
        # cups left by one. If there are no more cups left, set charge to 0.
        if self.coffeeCupsLeft > 0:
            self.coffeeCupsLeft -= 1

        if self.coffeeCupsLeft == 0:
            self.charge = 0


# create a list of zones, which define to what direction a given agant wants to go, as a function of the given agent's
# goal destination
zonelist = [Zone([48*15, 5*15, 68*15, 9*15], [0.707, 0.707]), Zone([48*15, 21*15, 68*15, 25*15], [0.707, -0.707]),
            Zone([-0.2*w, 8*15, 12.5*15, 22*15], [1, 0]), Zone([12.5*15, 8*15, 25*15, 22*15], [1, 0]),
            Zone([25*15, 8*15, 39.5*15+15, 22*15], [1, 0]), Zone([36.5*15-15, 8*15, 48*15, 22*15], [1, 0]),
            Zone([48*15, 8*15, 68*15, 22*15], [1, 0]), Zone([68*15, 8*15, 79.5*15, 22*15], [1, 0]),
            Zone([76.5*15, 8*15, 91*15, 22*15], [1, 0]), Zone([88*15, 8*15, 102*15, 22*15], [1, 0]),
            Zone([102*15, 8*15, 1.2*w, 22*15], [1, 0]), Zone([102*15, 8*15, w, 22*15], [1, 0]),
            Zone([0, 4.8*15, 3*15+15, 8*15], [0, 1]), Zone([35.5*15, 4.8*15, 40.5*15, 8*15], [0, 1]),
            Zone([88*15-15, 4.8*15, 91*15+15, 8*15], [0, 1]), Zone([25*15-15, 22*15, 28*15+15, 25.2*15], [0, -1]),
            Zone([75.5*15, 22*15, 80.5*15, 25.2*15], [0, -1]), Zone([113*15-15, 22*15, w, 25.2*15], [0, -1])]

# zone list for emergency situation
emergencyzones = [Zone([48*15, 5*15, 68*15, 9*15], [0.707, 0.707]), Zone([48*15, 21*15, 68*15, 25*15], [0.707, -0.707]),
                  Zone([-0.2*w, 8*15, 12.5*15, 22*15], [1, 0]), Zone([12.5*15, 8*15, 25*15, 22*15], [1, 0]),
                  Zone([25*15, 8*15, 39.5*15+15, 22*15], [1, 0]), Zone([36.5*15-15, 8*15, 48*15, 22*15], [1, 0]),
                  Zone([48*15, 8*15, 68*15, 22*15], [1, 0]), Zone([68*15, 8*15, 79.5*15, 22*15], [1, 0]),
                  Zone([76.5*15, 8*15, 91*15, 22*15], [1, 0]), Zone([88*15, 8*15, 102*15, 22*15], [1, 0]),
                  Zone([102*15, 8*15, 1.2*w, 22*15], [1, 0]), Zone([102*15, 8*15, w, 22*15], [1, 0]),
                  Zone([0, 4.8*15, 3*15+15, 8*15], [0, 1]), Zone([88*15-15, 4.8*15, 91*15+15, 8*15], [0, 1]),
                  Zone([25*15-15, 22*15, 28*15+15, 25.2*15], [0, -1]), Zone([113*15-15, 22*15, w, 25.2*15], [0, -1])]

# create a list of goal destinations to which an agent may want to go
goals = [(-0.3*w, 0.5*h), (1.3*w, 0.5*h), (0.015*w, 0.26*h), (0.328*w, 0.26*h), (0.772*w, 0.26*h), (0.987*w, 0.74*h),
         (0.672*w, 0.74*h), (0.23*w, 0.74*h)]

# convert goal destinations to np.arrays and scale them
goals = np.array(goals) / scale

# create coffee stands
coffeeStands = [coffeeStand([0.3*width, 0.35*width], [19*15/scale, 21*15/scale], imgpos=[0.345*w, 19.5*15]),
                coffeeStand([0.65*width, 0.70*width], [9*15/scale, 11*15/scale], imgpos=[0.695*w, 9.5*15])]
people_inline = [0, 0]  # people in line at coffee stands


def people_in_line_emergency_exits():
    # in an emergency situation, count how many persons who are currently using (i.e. are in the neighborhood) of each
    # of the two emergency exits

    global people_inline

    people_inline = [0, 0]
    cutoff = 16

    for i in range(len(people)):
        d = goals[0] - people[i].pos
        d = d.dot(d)
        if d < cutoff:
            people_inline[0] += 1
            continue

        d = goals[1] - people[i].pos
        d = d.dot(d)
        if d < cutoff:
            people_inline[1] += 1


def which_emergency_exit(person):
    # dynamically decide which emergency exit person wishes to use, based on a simple optimization algorithm that values
    # exits that are close in euclidian norm and exits that have few other agents in their neighborhood.

    inline_weight = 1.5

    d0 = goals[3] - person.pos
    d0 = np.sqrt(d0.dot(d0))
    d1 = goals[6] - person.pos
    d1 = np.sqrt(d1.dot(d1))

    w_goal_0 = d0 ** 3 + inline_weight * people_inline[0] ** 2
    w_goal_1 = d1 ** 3 + inline_weight * people_inline[1] ** 2
    if w_goal_1 < w_goal_0 or d1 < 2.5:
        return goals[6]
    else:
        return goals[3]


# emitter rate
lam_for = 0.0075

# create a list of emitters
emitters = [Emitter(xmin=-0.2*width, xmax=0, ymin=0.28*height, ymax=0.71*height, lam=3*lam_for),  # left
            Emitter(xmin=width, xmax=1.2*width, ymin=0.28*height, ymax=0.71*height, lam=3*lam_for),  # right
            Emitter(xmin=0, xmax=0.034*width-rad, ymin=0.16*height, ymax=0.25*height, lam=lam_for),  # lu
            Emitter(xmin=0.306*width+rad, xmax=0.349*width-rad, ymin=0.16*height, ymax=0.25*height, lam=lam_for),  # mu
            Emitter(xmin=0.75*width+rad, xmax=0.796*width-rad, ymin=0.16*height, ymax=0.25*height, lam=lam_for),  # ru
            Emitter(xmin=0.207*width+rad, xmax=0.25*width-rad, ymin=0.74*height, ymax=0.84*height, lam=lam_for),  # ld
            Emitter(xmin=0.65*width+rad, xmax=0.694*width-rad, ymin=0.74*height, ymax=0.84*height, lam=lam_for),  # md
            Emitter(xmin=0.966*width+rad, xmax=width, ymin=0.74*height, ymax=0.84*height, lam=lam_for)]  # rd

# print coffee stands (rectangles, Tekna/Nito logo and coffee count) on canvas
for stand in coffeeStands:
    rectangle = canvas.create_rectangle(((stand.pos_x[0]*scale, stand.pos_y[1]*scale),
                                         (stand.pos_x[1]*scale, stand.pos_y[0]*scale)), fill="#fff")
    stand.id = canvas.create_text(stand.imgpos[0], stand.imgpos[1], fill="black", font="Times 10 bold",
                                  text=str(stand.coffeeCupsLeft))

canvas.pack()

# initalized variables
ittr = 0                # counter for the number of elapsed iterations
QUIT = False
paused = False
step = False
circles = []            # for plotting person circles on canvas
velLines = []           # for plotting velocity lines on canvas
gvLines = []            # for plotting goal velocity lines on canvas
emergency = False       # boolean variable to signal if emergency situation
deleted_indexes = []    # array for keeping control of people to delete before next iteration
people = [None] * num   # initialise num persons


def initSim():
    # initial plotting of num number of people at iteration ittr = 0
    global rad, people

    print("")
    print("Simulation of Agents on Stripa.")
    print("Agents avoid collisions using prinicples based on the laws of anticipation seen in human pedestrians.")
    print("Green Arrow is Goal Velocity, Red Arrow is Current Velocity")
    print("SPACE to pause, 'S' to step frame-by-frame, 'V' to turn the velocity display on/off.")
    print("")

    for i in range(num):
        circles.append(canvas.create_oval(0, 0, rad, rad, fill="white"))
        velLines.append(canvas.create_line(0, 0, 10, 10, fill="red"))
        gvLines.append(canvas.create_line(0, 0, 10, 10, fill="green"))

        people[i] = Person(pos=[rnd.uniform(0, width), rnd.uniform(0.28*height, 0.71*height)],
                           vel=[rnd.uniform(-1, 1), rnd.uniform(-1, 1)], gv=[0.0, 0.0], goal=rnd.choice(goals))


def drawWorld():
    # function for plotting all people at the canvas at each iteration

    for i in range(len(people)):

        if people[i].coffeeCharge is 1:
            canvas.itemconfig(circles[i], fill="black")  # change color
        else:
            canvas.itemconfig(circles[i], fill="white")  # change color

        canvas.coords(circles[i], scale * (people[i].pos[0] - rad), scale * (people[i].pos[1] - rad),
                      scale * (people[i].pos[0] + rad), scale * (people[i].pos[1] + rad))
        canvas.coords(velLines[i], scale * people[i].pos[0], scale * people[i].pos[1],
                      scale * (people[i].pos[0] + 1. * rad * people[i].vel[0]),
                      scale * (people[i].pos[1] + 1. * rad * people[i].vel[1]))
        canvas.coords(gvLines[i], scale * people[i].pos[0], scale * people[i].pos[1],
                      scale * (people[i].pos[0] + 1. * rad * people[i].gv[0]),
                      scale * (people[i].pos[1] + 1. * rad * people[i].gv[1]))

        if drawVels:
            canvas.itemconfigure(velLines[i], state="normal")
            canvas.itemconfigure(gvLines[i], state="normal")
        else:
            canvas.itemconfigure(velLines[i], state="hidden")
            canvas.itemconfigure(gvLines[i], state="hidden")


def update_goal_velocity(person, index):
    global deleted_indexes

    # the function updates person's goal velocity based on his goal and the zone he's currently in

    zone = which_zone(person)  # find which zone he is in

    if zone is None:  # If person is in no defined zone, delete the person
        deleted_indexes.append(index)
        return

    goal = person.goal  # retrive his goal destination

    if goal[0] < zone.pos[0]:  # goal is to the left of the zone
        person.gv = gv_speed*np.array([-zone.field[0], zone.field[1]])
    elif goal[0] > zone.pos[2]:  # goal is to the right of the zone
        person.gv = gv_speed*np.array([zone.field[0], zone.field[1]])
    else:
        # the goal is either above the current zone, or in the current zone

        # if the ultimate goal is in the current zone, delete person
        if zone.pos[0] < person.goal_save[0] < zone.pos[2] and zone.pos[1] < person.goal_save[1] < zone.pos[3]:
            deleted_indexes.append(index)
            return

        # if the goal is above the current zone, move directly towards the goal destination
        p = person.goal - person.pos
        person.gv = p / np.sqrt(p.dot(p)) * gv_speed


def which_zone(person):
    # find which zone person is currently in
    [x, y] = person.pos
    for zone in zonelist:
        if zone.pos[0] < x < zone.pos[2] and zone.pos[1] < y < zone.pos[3]:
            return zone

    # if person is in no defined zone, return None
    return None


def findNeighbors():
    # find all neighboring actors and compute their corresponding time-to-collision
    global people

    for i in range(len(people)):
        people[i].neighbors = []
        people[i].nt = []
        vel_angle = np.arctan2(people[i].vel[1], people[i].vel[0])
        for j in range(len(people)):
            if i == j:
                continue

            d = people[i].pos - people[j].pos
            d_angle = np.arctan2(d[1], d[0])
            l2 = d.dot(d)
            s2 = sight ** 2
            if l2 < s2 and abs(d_angle - vel_angle) > np.pi / 2:
                people[i].neighbors.append(j)
                people[i].nt.append(sqrt(l2))


def dE(persona, personb):
    # compute the interaction force between two neighbors, and return it

    p = personb.pos - persona.pos  # relative position
    v = persona.vel - personb.vel  # relative velocity
    dist = sqrt(p.dot(p))          # distance between neighbors
    r = rad                        # temporary radius
    if dist < 2 * rad:             # shrink overlapping agents
        r = dist/2.001

    a = v.dot(v)
    b = p.dot(v)
    c = p.dot(p) - 4*r*r
    discr = b * b - a * c

    if discr < 0 or -0.001 < a < 0.001:
        return np.array([0, 0])

    discr = sqrt(discr)
    t = (b - discr) / a

    if t < 0 or t > 999:
        return np.array([0, 0])

    d = k * exp(-t / t0) * (v - (v * b - p * a) / (discr)) / (a * t ** m) * (m / t + 1 / t0)
    return d


def closest_point_line_segment(c, wall):
    # find the closest point on a line segment (wall) from a person's centre, c

    line_start = wall[0:2]
    line_end = wall[2:4]
    dota = (c - line_start).dot(line_end - line_start)
    if dota <= 0:
        return line_start
    dotb = (c - line_end).dot(line_start - line_end)
    if dotb <= 0:
        return line_end
    slope = dota / (dota + dotb)
    return line_start + (line_end - line_start) * slope


def normal(wall):
    # compute normal vector of a wall

    p = wall[2:4] - wall[0:2]
    norm = np.array([-p[1], p[0]])
    return norm / np.sqrt(norm.dot(norm))


def wallforces(person):
    # compute and return wall forces acting on a person

    global walllist, rad

    F = [0, 0]

    for wall in walllist:

        # find closest point to given wall, if too far away, do not care about given wall
        closest = closest_point_line_segment(person.pos, wall) - person.pos
        dw = closest.dot(closest)
        if dw > sight:
            continue

        r = np.sqrt(dw) if dw < rad ** 2 else rad

        t_min = 3

        discCollision = 0
        segmentCollision = 0

        a = person.vel.dot(person.vel)

        # does particle collide with top capsule
        w_temp = wall[0:2] - person.pos
        b_temp = w_temp.dot(person.vel)
        c_temp = w_temp.dot(w_temp) - r ** 2
        discr_temp = b_temp * b_temp - a * c_temp
        if discr_temp > 0 and abs(a) > 0:
            discr_temp = sqrt(discr_temp)
            t = (b_temp - discr_temp) / a
            if 0 < t < t_min:
                t_min = t
                b = b_temp
                discr = discr_temp
                w = w_temp
                discCollision = 1

        # does particle collide with bottom capsule
        w_temp = wall[2:4] - person.pos
        b_temp = w_temp.dot(person.vel)
        c_temp = w_temp.dot(w_temp) - r ** 2
        discr_temp = b_temp * b_temp - a * c_temp
        if discr_temp > 0 and abs(a) > 0:
            discr_temp = sqrt(discr_temp)
            t = (b_temp - discr_temp) / a
            if 0 < t < t_min:
                t_min = t
                b = b_temp
                discr = discr_temp
                w = w_temp
                discCollision = 1

        # does particle collide with line segment from the front
        w1 = wall[0:2] + r * normal(wall)
        w2 = wall[2:4] + r * normal(wall)
        w_temp = w2 - w1
        D = np.cross(person.vel, w_temp)
        if D != 0:
            t = np.cross(w_temp, person.pos - w1) / D
            # s = (p+velocity*t-o1_temp).dot(o_temp)/(o_temp.dot(o_temp))
            s = np.cross(person.vel, person.pos - w1) / D
            if 0 < t < t_min and 0 <= s <= 1:
                t_min = t
                w = w_temp
                discCollision = 0
                segmentCollision = 1

        # does particle collide with line segment from the bottom
        w1 = wall[0:2] - r * normal(wall)
        w2 = wall[2:4] - r * normal(wall)
        w_temp = w2 - w1
        D = np.cross(person.vel, w_temp)
        if D != 0:
            t = np.cross(w_temp, person.pos - w1) / D
            # s = (p + velocity * t - o1_temp).dot(o_temp) / (o_temp.dot(o_temp))
            s = np.cross(person.vel, person.pos - w1) / D
            if 0 < t < t_min and 0 <= s <= 1:
                t_min = t
                w = w_temp
                discCollision = 0
                segmentCollision = 1

        # compute forces acting on the particle
        if discCollision:
            FAvoid = -k * np.exp(-t_min / t0) * (person.vel - (b * person.vel - a * w) / discr) / (a * (t_min ** m)) * (
                    m / t_min + 1 / t0)
            F += FAvoid
        if segmentCollision:
            FAvoid = k * np.exp(-t_min / t0) / (t_min ** m * np.cross(person.vel, w)) * (m / t_min + 1 / t0) * np.array(
                [-w[1], w[0]])
            F += FAvoid
    return F


def hardwall(i, dt, a):
    # computes a hard wall computation, where people who will collide with a wall ... isn't allowed to do so

    global people

    p = people[i].pos + (a * dt) * dt  # new position of person with index i if he where to continue with acceleration a
    r = rad

    for wall in walllist:

        q = closest_point_line_segment(people[i].pos, wall)  # find closest point on the wall to person's centre
        y = (people[i].pos - q).dot(people[i].pos - q)  # relative position between person and wall point q
        if y <= rad ** 2:
            # if person already as collided (intersects) with the given wall, shrink the persons radius
            r = np.sqrt(y)/1.005

        q = closest_point_line_segment(p, wall)  # find closest point on the wall to person's would be new position p
        if (p - q).dot(p - q) <= r ** 2:
            # if the new would be position q assures that the person intersects with the wall, give the person a new
            # velocity vector similar to an "elastisk stöt"

            w = wall[2:4] - wall[0:2]
            n = np.array([-w[1], w[0]])
            u = people[i].vel.dot(n) / n.dot(n) * n
            people[i].vel += -2 * u                  # update person's velocity
            people[i].pos += people[i].vel * dt      # update person's position
            return

    # if person doesn't collide with any walls, update position and velocity according to original acceleration a
    people[i].vel += a * dt
    people[i].pos += people[i].vel * dt


def find_closest_coffeeStand(person):
    # for a person who wants coffee, find the closest coffee stand

    epsilon = width*0.1

    # iterate over all stands
    for stand in coffeeStands:

        r = stand.pos - person.pos  # relative position between person and stand
        dist = np.sqrt(r.dot(r))    # squared norm of the relative distance

        # if a stand is sufficiently close and it has coffee cups left (and sufficiently few persons are in line at the
        # stand)
        if dist < epsilon and stand.coffeeCupsLeft and (stand.inline < 5 or person.inline):

            # if a person is not already in line at the coffee stand
            if not person.inline:
                stand.inline += 1     # update counter for the number in line at the given stand
                person.inline = True  # person is in a coffee stand line
                return stand.pos      # return the stand's position as the persons new goal destination

            # if person has reached the stand and gotten a cup of coffee
            if stand.pos_x[0] < person.pos[0] < stand.pos_x[1] and stand.pos_y[0] < person.pos[1] < stand.pos_y[1]:
                person.coffeeCharge = 0  # person doesn't want any more coffee
                person.inline = False    # person is no longer in line at a stand
                stand.inline -= 1        # one less person in line at the stand
                stand.studentCoffee()    # one less cup of coffee available at the stand

                # draw number of coffee cups left at the canvas
                canvas.delete(stand.id)
                stand.id = canvas.create_text(stand.imgpos[0], stand.imgpos[1], fill="black", font="Times 10 bold", text=str(stand.coffeeCupsLeft))

                #  return the person's original goal destination as the persons "new" goal destination
                return person.goal_save

            return stand.pos  # else, return the stand's position as the persons new goal destination

    return person.goal_save  # else, return the person's original goal destination as the persons "new" goal destination


def update(dt):
    global people, coffeeStands, emergency, deleted_indexes, zonelist

    # update people's position, velocity, goal velocity (and possibly coffee charge)

    findNeighbors()       # find all pairwise neighbors
    F = []                # initialise force
    deleted_indexes = []  # initialise deleted indexes array

    for i in range(len(people)):
        F.append(np.zeros(2))

    # Emergency situation: if there are no cups of coffee left at Tekna stand, initiate emergency situation
    if coffeeStands[0].coffeeCupsLeft is 0 or coffeeStands[1].coffeeCupsLeft is 0:
        canvas.create_text(w/2, h/2, fill="red", font="Times 50 bold", text="EMERGENCY!")
        emergency = True

    if emergency is True:
        zonelist = emergencyzones

    for i in range(len(people)):
        # update persons goal velocity and compute force component from goal velocity and from stochastic force
        update_goal_velocity(people[i], i)
        F[i] += (people[i].gv - people[i].vel) / .5
        F[i] += 1 * np.array([rnd.uniform(-1.5, 1.5), rnd.uniform(-1.5, 1.5)])

        # compute force component from neighbors
        for n, j in enumerate(people[i].neighbors):
            F[i] += -dE(people[i], people[j])

        # compute force component from walls
        F[i] += wallforces(people[i])

        # if not an emergency situation and person wants coffee, check if there is a coffee stand nearby. If there is,
        # change (temporary) goal to coffee stand
        if emergency:
            people[i].goal = which_emergency_exit(people[i])

        if not emergency and people[i].coffeeCharge:
            people[i].goal = find_closest_coffeeStand(people[i])

    for i in range(len(people)):
        # if acceleration of person i is too large, scale it down
        a = F[i]
        mag = np.sqrt(a.dot(a))
        if mag > maxF:
            a = maxF * a / mag

        hardwall(i, dt, a)  # change velocity and position of person i (avoiding collision with walls)


def on_key_press(event):
    global paused, step, QUIT, drawVels
    if event.keysym == "space":
        paused = not paused
    if event.keysym == "s":
        step = True
        paused = False
    if event.keysym == "v":
        drawVels = not drawVels
    if event.keysym == "Escape":
        QUIT = True


def drawFrame(dt=0.05):
    # the main simulation loop where new people are emitted, people are deleted, the update function update() is called
    # and each iteration is drawn on canvas

    global step, paused, ittr, deleted_indexes, circles, velLines, gvLines, people

    # Simulation Loop
    if ittr >= maxIttr or QUIT:
        print("%s itterations ran ... quitting" % ittr)
        win.destroy()
    else:
        if not paused:

            # delete people (that are outside of any defined zones or have reached their goal destination)
            deleted_indexes = np.flip(np.sort(deleted_indexes), 0)
            for i in deleted_indexes:
                canvas.delete(circles[i])
                canvas.delete(velLines[i])
                canvas.delete(gvLines[i])
                people.pop(i)
                circles.pop(i)
                velLines.pop(i)
                gvLines.pop(i)

            # emit new people: number of people to be spawned at the given emitter at the given time is drawn (at t=0)
            # from a Poisson process.
            for emitter in emitters:
                if emitter.spawn[ittr]:
                    for k in range(emitter.spawn[ittr]):

                        # if not an emergency situation
                        if not emergency:

                            # to what goal does the person want to go?
                            choice = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7],
                                                      p=[0.4, 0.4, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04])

                            # create the person, one third of all persons wants coffee
                            p = Person(
                                pos=[rnd.uniform(emitter.xmin, emitter.xmax), rnd.uniform(emitter.ymin, emitter.ymax)],
                                goal=goals[choice], coffee=rnd.choice([0, 0, 1]))

                        # if an emergency situation, exit through one of two exit goals; no one wants coffee
                        else:
                            p = Person(
                                pos=[rnd.uniform(emitter.xmin, emitter.xmax), rnd.uniform(emitter.ymin, emitter.ymax)],
                                goal=goals[0])

                        printp = True

                        # check if person p is emitted on top of another person, in that case, delete p
                        for i in range(len(people)):
                            d = people[i].pos - p.pos
                            if np.sqrt(d.dot(d)) < 2 * rad:
                                printp = False
                                break

                        # if p isn't emitted on top of another person, print p and append to person list
                        if printp:
                            people.append(p)

                            if p.coffeeCharge is 1:
                                circles.append(canvas.create_oval(0, 0, rad, rad, fill="black"))
                            else:
                                circles.append(canvas.create_oval(0, 0, rad, rad, fill="white"))

                            velLines.append(canvas.create_line(0, 0, 10, 10, fill="red"))
                            gvLines.append(canvas.create_line(0, 0, 10, 10, fill="green"))



            update(dt)  # update people's position, velocity, goal velocity and much more
            ittr += 1

        drawWorld()  # draw the iteration frame on canvas
        if step is True:
            step = False
            paused = True

        win.title("Simulering av Stripa")
        win.after(framedelay, drawFrame)


# run the simulation
win.bind("<space>", on_key_press)
win.bind("s", on_key_press)
win.bind("<Escape>", on_key_press)
win.bind("v", on_key_press)
initSim()
start_time = time.time()
win.after(framedelay, drawFrame)
mainloop()