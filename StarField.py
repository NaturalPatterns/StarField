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

#------------------------------------------------------------
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=51, help="seed for the RNG")
parser.add_argument("--fps", type=int, default=25, help="frames per second")
parser.add_argument("--N", type=int, default=10000, help="number of particles")
parser.add_argument("--size", type=float, default=10, help="size of symbols")
parser.add_argument("--V", type=float, default=1, help="speed")
parser.add_argument("--mag", type=float, default=5, help="magnification")
parser.add_argument("--bound_width", type=float, default=4, help="speed")
parser.add_argument("--bound_depth", type=float, default=20, help="speed")
parser.add_argument("--T", type=float, default=3., help="duration")
parser.add_argument("--d_min", type=float, default=1.e-6, help="min distance")
parser.add_argument("--d_max", type=float, default=6., help="max distance")
parser.add_argument("--theta", type=float, default=np.pi/64, help="angle of view wrt displacement")
parser.add_argument("--fname", type=str, default=None, help="filename to save the animation to")
parser.add_argument("--verbose", type=bool, default=True, help="Displays more verbose output.")

opt = parser.parse_args()

if opt.verbose:
    print(opt)
#------------------------------------------------------------

import sys
BACKEND = 'tkagg'
import matplotlib
matplotlib.use(BACKEND)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ParticleBox:
    """ StarField

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self, opt):
        bounds = [-opt.bound_width, opt.bound_width, -opt.bound_width, opt.bound_width, 0, opt.bound_depth]
        self.bounds = np.asarray(bounds, dtype=float)
        self.opt = opt
        self.time_elapsed = 0.
        self.init()
        self.project()

    def init(self):
        """
        Generate a starfield in a box defined by bounds

        X, Y, Z, R, G, B, alpha
        """

        self.pos = np.random.rand(self.opt.N, 7)
        for i in range(3):
            self.pos[:, i] *= (self.bounds[2*i+1] - self.bounds[2*i])
            self.pos[:, i] += self.bounds[2*i]

        # Star colors http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html http://www.vendian.org/mncharity/dir3/starcolor/
        O3 = np.array([144., 166., 255.])
        O3 /= 255.
        self.pos[:, 3:-1] = O3[None, :]
        M4Ia = np.array([255., 185., 104.])
        M4Ia /= 255.
        self.pos[np.random.rand(self.opt.N)>.5, 3:-1] = M4Ia[None, :]

        self.pos[:, -1] = .8 + .2*self.pos[:, -1]

    def project(self):
        """
        Project positions on the screen

        """
        # update positions compared to observer
        pos = self.pos.copy()

        # center coordinates around obs coords
        pos[:, 0] -= np.sin(self.opt.theta) * self.opt.V * self.time_elapsed
        pos[:, 2] -= np.cos(self.opt.theta) * self.opt.V * self.time_elapsed

        # wrap in a novel box around obs coords
        for i in range(3):
            pos[:, i] = self.bounds[2*i] + np.mod(pos[:, i], self.bounds[2*i + 1]-self.bounds[2*i])

        d = (pos**2).sum(axis=1)**.5
        # order according to depth
        ind_sort = np.argsort(d)
        pos = pos[ind_sort, :]
        d = (pos**2).sum(axis=1)**.5

        # ind_visible = (pos[:, 2] > 0) * (self.d_min<d) * (d<self.d_max)
        ind_visible = (pos[:, 2] > self.opt.d_min) * (d < self.opt.d_max)
        N_visible = int(np.sum(ind_visible))

        # self.state = [X, Y, size, R, G, B, alpha]
        self.state = np.ones((N_visible, 7))
        for i in range(2):
            self.state[:, i] = self.opt.mag * pos[ind_visible, i] / pos[ind_visible, 2]

        self.state[:, 2] = self.opt.size / d[ind_visible]

        # colors do not change
        self.state[:, 3:] = pos[ind_visible, 3:]


    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        self.project()


# set up initial state
np.random.seed(opt.seed)
box = ParticleBox(opt)
dt = 1. / opt.fps # 30fps
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
    if box.time_elapsed > opt.T: sys.exit()
    return ax

ani = animation.FuncAnimation(fig, animate, frames=int(opt.T*opt.fps), interval=1000/opt.fps)
if not opt.fname is None:
    ani.save(opt.fname + '.mp4', fps=fps, extra_args=['-vcodec', 'libx264'], savefig_kwargs=dict( facecolor='black'), dpi=300)
    # import os
    # os.system('ffmpeg -i starfield.mp4  starfield.gif')

plt.show()
