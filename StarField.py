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
parser.add_argument("--noise", type=float, default=.005, help="diffusion aprameter of the brownian motion of particles")
parser.add_argument("--size", type=float, default=10, help="size of symbols")
parser.add_argument("--radius", type=float, default=.2, help="radius of center blind spot")
parser.add_argument("--fix_size", type=float, default=40, help="size of fixation symbol")
parser.add_argument("--V", type=float, default=1, help="speed")
parser.add_argument("--A", type=float, default=0, help="acceleration")
parser.add_argument("--mag", type=float, default=5, help="magnification")
parser.add_argument("--bound_width", type=float, default=4, help="speed")
parser.add_argument("--bound_depth", type=float, default=20, help="speed")
parser.add_argument("--T", type=float, default=3., help="duration")
parser.add_argument("--tau", type=float, default=.5, help="mean kill duration")
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
if False:
    BACKEND = 'Qt5Agg'
else:
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


    def random_positions(self):
        pos = np.random.rand(self.opt.N, 3)
        for i in range(3):
            pos[:, i] *= (self.bounds[2*i+1] - self.bounds[2*i])
            pos[:, i] += self.bounds[2*i]
        return pos

    def init(self):
        """
        Generate a starfield in a box defined by bounds

        X, Y, Z, R, G, B, alpha
        """

        self.pos = np.random.rand(self.opt.N, 7)
        self.pos[:, :3] = self.random_positions()

        # Star colors http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html http://www.vendian.org/mncharity/dir3/starcolor/
        O3 = np.array([144., 166., 255.])
        O3 /= 255.
        self.pos[:, 3:-1] = O3[None, :]
        M4Ia = np.array([255., 185., 104.])
        M4Ia /= 255.
        self.pos[np.random.rand(self.opt.N)>.5, 3:-1] = M4Ia[None, :]

        self.pos[:, -1] = .8 + .2*self.pos[:, -1]

    def project(self, dt=0.):
        """
        Project positions on the screen

        """
        # add noise to the trajectory
        self.pos[:, :3] += np.sqrt(dt) * self.opt.noise * np.random.randn(self.opt.N, 3)

        # kill some particles and place them again at random in the box
        if dt >0. :
            ind_kill = np.random.rand(self.opt.N) < 1 - np.exp( - dt / self.opt.tau)
            # print( dt, 1 - np.exp( - dt / self.opt.tau), (ind_kill).sum())
            ind_kill = ind_kill[:, None] * np.array([True]*3)
            self.pos[:, :3] = np.where(ind_kill, # condition
                                       self.random_positions(), # kill if True
                                       self.pos[:, :3] # keep if False
                                       )

        # update positions compared to observer
        pos = self.pos.copy()

        # center coordinates around obs coords
        x = self.opt.V * self.time_elapsed + 1/2 *self.opt.A * self.time_elapsed ** 2
        pos[:, 0] -= np.sin(self.opt.theta) * x
        pos[:, 2] -= np.cos(self.opt.theta) * x

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
        self.project(dt)


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
    if box.opt.radius > 0:
        circle = plt.Circle((0,0), box.opt.radius, color='k')
        ax.add_artist(circle)
    fixation = ax.scatter([0], [0], marker='+', c='white', s=box.opt.fix_size, zorder=2)
    ax.set_xlim(-ratio, ratio)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    if box.time_elapsed > opt.T: sys.exit()
    return ax

ani = animation.FuncAnimation(fig, animate, frames=int(opt.T*opt.fps), interval=1000/opt.fps)
if not opt.fname is None:
    ani.save(opt.fname + '.mp4', fps=opt.fps, extra_args=['-vcodec', 'libx264'], savefig_kwargs=dict( facecolor='black'), dpi=300)
    # import os
    # os.system('ffmpeg -i starfield.mp4  starfield.gif')

plt.show()
