"""
Star field

adapted from
Animation of Elastic collisions with Gravity
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class ParticleBox:
    """ StarField

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 bounds = [0, 1, 0, 1, 0, 200],
                 size = 10.,
                 N = 1000,
                 V = 2.,
                 theta = 0.):
        self.bounds = np.asarray(bounds, dtype=float)
        self.N = N
        self.time_elapsed = 0
        self.V = V
        self.size = size
        self.theta = theta
        self.d_max = 20
        self.d_min = .1
        self.init()
        self.project()

    def init(self):
        """
        Generate a starfield in a box defined by bounds

        """
        self.pos = np.random.rand(self.N, 3)
        for i in range(3):
            self.pos[:, i] *= (self.bounds[2*i+1] - self.bounds[2*i])
            self.pos[:, i] -= self.bounds[2*i]
            print(self.pos[:, i].min(), self.pos[:, i].max())

    def project(self):
        """
        Project positions on the screen

        """
        # update positions compared to observer
        pos = self.pos.copy()
        # HACK
        pos[:, :2] *= 2
        pos[:, :2] -= 1
        pos[:, 0] -= np.sin(self.theta) * self.V * self.time_elapsed
        pos[:, 2] -= np.cos(self.theta) * self.V * self.time_elapsed
        # TODO: wrap all in a novel box

        d = (pos**2).sum(axis=1)**.5
        ind_visible = (pos[:, 2] > 0) * (self.d_min<d) * (d<self.d_max)

        #self.state = [X, Y, ms]
        N_visible = int(np.sum(ind_visible))
        self.state = np.ones((N_visible, 3))
        # print (self.time_elapsed, N_visible, self.state[:, 1].shape, pos[ind_visible, 1].shape, d[ind_visible].shape)#, d[ind_visible])
        self.state[:, 0] = pos[ind_visible, 0] / d[ind_visible]
        self.state[:, 1] = pos[ind_visible, 1] / d[ind_visible]
        self.state[:, 2] = self.size / d[ind_visible]

        # ind_visible = (pos[:, 2] > 0) * (d<self.d_max)

        # self.state = np.random.rand(100, 3)
        # for i in range(3):
        #     self.state[:, i] *= (self.bounds[2*i+1] - self.bounds[2*i])
        #     self.state[:, i] -= self.bounds[2*i]

        # HACK
        self.state /= 2
        self.state += .5


    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        self.project()

#------------------------------------------------------------
# set up initial state
np.random.seed(42)
box = ParticleBox()
dt = 1. / 30 # 30fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, 1), ylim=(0, 1))

# particles holds the locations of the particles
# particles, = ax.plot([], [], 'bo', ms=6)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

# box.step(dt)
# ax.scatter(box.state[:, 0], box.state[:, 1], marker='o', c='b', s=6)
#
# def init():
#     """initialize animation"""
#     global box, rect
#     particles, = ax.scatter(box.state[:, 0], box.state[:, 1], 'bo', ms=6)
#     rect.set_edgecolor('none')
#     return particles, rect
#
def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)
    ax.cla()

    # ms = [int(k* box.size * fig.dpi * 2 * box.size * fig.get_figwidth()
    #          / np.diff(ax.get_xbound())[0]) for  k in box.state[:, 2]]
    # ms = [int(k* box.size) for k in box.state[:, 2] ]
    ms = 6

    # update pieces of the animation
    rect.set_edgecolor('k')
    # print(len(ms), box.state.shape)
    # print(box.state[:, 0].min(), box.state[:, 0].max())
    particles = ax.scatter(box.state[:, 0], box.state[:, 1], marker='o', c='b', s=box.state[:, 2])
    #particles.set_data(box.state[:, 0], box.state[:, 1])
    #particles.set_markersize(box.state[:, 2]*ms)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True)#, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('starfield.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
