import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1, 1],
                               [-0.5, 0.5, 0.5, 0.5, 1],
                               [-0.5, -0.5, -0.5, 0.5, -1]],
                 bounds = [-2, 2, -2, 2],
                 size = 0.04,
                 M = 0.05):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.population = [[Ac_number],
                           [0],
                           [(len(self.state)-Ac_number)]]

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        numA = 0
        numAB = 0
        numB = 0

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            #print(i1, i2)
            #print("crashed")
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:4]
            v2 = self.state[i2, 2:4]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:4] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:4] = v_cm - v_rel * m1 / (m1 + m2)

            status1 = self.state[i1][4]
            status2 = self.state[i2][4]

            #print(self.state[:, 4])
            #Ac-AB
            if (self.state[i1][4] == -1) and (self.state[i2][4] == 0.5):
                #print("here1")
                self.state[i2][4] = 0
                numAB += -1
                numA += 1
            elif (self.state[i1][4] == 0.5) and (self.state[i2][4] == -1):
                #print("here2")
                self.state[i1][4] = 0
                numAB += -1
                numA += 1
            #Ac-B
            elif (self.state[i1][4] == -1) and (self.state[i2][4] == 1):
                #print("here3")
                self.state[i2][4] = 0.5
                numB += -1
                numAB += 1
            elif (self.state[i1][4] == 1) and (self.state[i2][4] == -1):
                #print("here4")
                self.state[i1][4] = 0.5
                numB += -1
                numAB += 1
            #A-AB
            elif (self.state[i1][4] == 0) and (self.state[i2][4] == 0.5):
                #print("here5")
                self.state[i2][4] = 0
                numAB += -1
                numA += 1
            elif (self.state[i1][4] == 0.5) and (self.state[i2][4] == 0):
                #print("here6")
                self.state[i1][4] = 0
                numAB += -1
                numA += 1
            #A-B
            elif (self.state[i1][4] == 1) and (self.state[i2][4] == 0):
                #print("here7")
                self.state[i1][4] = 0.5
                self.state[i2][4] = 0.5
                numA -= 1
                numB -= 1
                numAB += 2
            elif (self.state[i1][4] == 0) and (self.state[i2][4] == 1):
                #print("here8")
                self.state[i1][4] = 0.5
                self.state[i2][4] = 0.5
                numA -= 1
                numB -= 1
                numAB += 2
            #AB-B
            elif (self.state[i1][4] == 0.5) and (self.state[i2][4] == 1):
                #print("here9")
                self.state[i1][4] = 1
                numAB += -1
                numB += 1
            elif (self.state[i1][4] == 1) and (self.state[i2][4] == 0.5):
                #print("here10")
                self.state[i2][4] = 1
                numAB += -1
                numB += 1

            #print(numA, numAB, numB)
            #print(self.state[:, 4])

        self.population[0].append(self.population[0][-1]+numA)
        self.population[1].append(self.population[1][-1]+numAB)
        self.population[2].append(self.population[2][-1]+numB)

        # update positions
        self.state[:, :2] += dt * self.state[:, 2:4]

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

#------------------------------------------------------------
# set up initial state
particlenumber = 1000
particlesize = 0.02
box_length = 2
Ac_number = 80

np.random.seed()
init_state = -0.5 + np.random.random((particlenumber, 5))
init_state[:, :2] *= (box_length*2)*0.95
init_state[:Ac_number, 4:] = -1
init_state[Ac_number:, 4:] = 1

init_state[0][4] = -1
init_state[1][4] = 1
init_state[2][4] = 0.5

box = ParticleBox(init_state, size=particlesize, bounds = [-box_length, box_length, -box_length, box_length])
dt = 1. / 30 # 30fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure(figsize=(7,7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=True,
                     xlim=(-(box_length+0.5), box_length+0.5), ylim=(-(box_length+0.5), box_length+0.5))

# particles holds the locations of the particles
particlesc, = ax.plot([], [], 'ko', ms=6)
particles, = ax.plot([], [], 'ro', ms=6)
particlesAB, = ax.plot([], [], 'mo', ms=6)
particlesB, = ax.plot([], [], 'bo', ms=6)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')

    Ac_array = []
    A_array = []
    AB_array = []
    B_array = []


    for i in range(len(box.state)):
        if box.state[i][4] == -1 :
            Ac_array.append(box.state[i])
        elif box.state[i][4] == 0:
            A_array.append(box.state[i])
        elif (box.state[i][4] == 0.5) :
            AB_array.append(box.state[i])
        else :
            B_array.append(box.state[i])

    particlesc, = ax.plot([], [], 'ro', ms=6)
    particles, = ax.plot([], [], 'ro', ms=6)
    particlesAB, = ax.plot([], [], 'mo', ms=6)
    particlesB, = ax.plot([], [], 'bo', ms=6)

    Ac_array = np.asarray(Ac_array)
    A_array = np.asarray(A_array)
    AB_array = np.asarray(AB_array)
    B_array = np.asarray(B_array)

    if (len(Ac_array) > 0):
        particlesc.set_data(Ac_array[:, 0], Ac_array[:, 1])
        particlesc.set_markersize(ms)
        particlesc.set_markerfacecolor("red")

    if (len(A_array) > 0):
        particles.set_data(A_array[:, 0], A_array[:, 1])
        particles.set_markersize(ms)
        particles.set_markerfacecolor("red")

    if (len(AB_array) > 0):
        particlesAB.set_data(AB_array[:, 0], AB_array[:, 1])
        particlesAB.set_markersize(ms)
        particlesAB.set_markerfacecolor("magenta")

    if (len(B_array) > 0):
        particlesB.set_data(B_array[:, 0], B_array[:, 1])
        particlesB.set_markersize(ms)
        particlesB.set_markerfacecolor("blue")

    return particlesc, particles, particlesAB, particlesB, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)

#plt.subplot(122)
#plt.plot([3,3,3])

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

print("Total = ", particlenumber, "Ac = ", Ac_number)

plt.show()