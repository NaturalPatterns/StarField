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
import sys
BACKEND = 'tkagg'
import matplotlib
matplotlib.use(BACKEND)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ParticleBox:
    """ StarField

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 bounds = [-4, 4, -4, 4, 0, 20],
                 size = 30.,
                 N = 10000,
                 V = 1.,
                 theta = np.pi/64):
        self.bounds = np.asarray(bounds, dtype=float)
        self.N = N
        self.time_elapsed = 0
        self.V = V
        self.size = size
        self.theta = theta
        self.d_max = 6.
        self.d_min = 1.e-6
        self.mag = 5.
        self.T = 3.
        self.init()
        self.project()

    def init(self):
        """
        Generate a starfield in a box defined by bounds

        X, Y, Z, R, G, B, alpha
        """

        self.pos = np.random.rand(self.N, 7)
        for i in range(3):
            self.pos[:, i] *= (self.bounds[2*i+1] - self.bounds[2*i])
            self.pos[:, i] += self.bounds[2*i]

        # Star colors http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html http://www.vendian.org/mncharity/dir3/starcolor/
        O3 = np.array([144., 166., 255.])
        O3 /= 255.
        self.pos[:, 3:-1] = O3[None, :]
        M4Ia = np.array([255., 185., 104.])
        M4Ia /= 255.
        self.pos[np.random.rand(self.N)>.5, 3:-1] = M4Ia[None, :]

        self.pos[:, -1] = .8 + .2*self.pos[:, -1]

    def project(self):
        """
        Project positions on the screen

        """
        # update positions compared to observer
        pos = self.pos.copy()

        # center coordinates around obs coords
        pos[:, 0] -= np.sin(self.theta) * self.V * self.time_elapsed
        pos[:, 2] -= np.cos(self.theta) * self.V * self.time_elapsed

        # wrap in a novel box around obs coords
        for i in range(3):
            pos[:, i] = self.bounds[2*i] + np.mod(pos[:, i], self.bounds[2*i + 1]-self.bounds[2*i])

        d = (pos**2).sum(axis=1)**.5
        # order according to depth
        ind_sort = np.argsort(d)
        pos = pos[ind_sort, :]
        d = (pos**2).sum(axis=1)**.5

        # ind_visible = (pos[:, 2] > 0) * (self.d_min<d) * (d<self.d_max)
        ind_visible = (pos[:, 2] > self.d_min) * (d < self.d_max)
        N_visible = int(np.sum(ind_visible))

        # self.state = [X, Y, size, R, G, B, alpha]
        self.state = np.ones((N_visible, 7))
        for i in range(2):
            self.state[:, i] = self.mag * pos[ind_visible, i] / pos[ind_visible, 2]

        self.state[:, 2] = self.size / d[ind_visible]

        # colors do not change
        self.state[:, 3:] = pos[ind_visible, 3:]


    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        self.project()

#------------------------------------------------------------
# set up initial state
np.random.seed(42)
box = ParticleBox()
fps = 30 # 30fps
dt = 1. / fps # 30fps
figsize = (15, 8)
ratio = figsize[0]/figsize[1]
#------------------------------------------------------------
# set up figure and animation
fig, ax = plt.subplots(facecolor='black', subplot_kw=dict(autoscale_on=False))
fig.set_facecolor('black')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

def animate(i):
    """perform animation step"""
    global box, dt, ax, fig
    box.step(dt)
    ax.cla()
    # note: s is the marker size in points**2.
    particles = ax.scatter(box.state[:, 0], box.state[:, 1], marker='*', c=box.state[:, 3:], s=box.state[:, 2]**2, zorder=1)

    ax.set_xlim(-ratio, ratio)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    if box.time_elapsed > box.T: sys.exit()
    return ax

ani = animation.FuncAnimation(fig, animate, frames=int(box.T*fps), interval=1000/fps)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('starfield.mp4', fps=fps, extra_args=['-vcodec', 'libx264'], savefig_kwargs=dict( facecolor='black'), dpi=300)
# import os
# os.system('ffmpeg -i starfield.mp4  starfield.gif')

plt.show()
