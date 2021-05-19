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
parser.add_argument("--dpi", type=int, default=300, help="dots per inch")
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
parser.add_argument("--marker", type=str, default='*', help="marker to use")
parser.add_argument("--facecolor", type=str, default='black', help="facecolor to use")
parser.add_argument("--fname", type=str, default=None, help="filename to save the animation to")
parser.add_argument("--vext", type=str, default='mp4', help="video MIME type")
parser.add_argument("--backend", type=str, default='Agg', help="pyplot backend")
parser.add_argument("--verbose", default=False, action='store_true', help="Displays more verbose output.")
parser.add_argument("--realistic", default=False, action='store_true', help="Displays a realistic looking output.")

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
matplotlib.use(opt.backend)

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

        # euclidian distance
        d = (pos**2).sum(axis=1)**.5
        # order according to depth
        ind_sort = np.argsort(d)
        pos = pos[ind_sort, :]
        # recompute euclidian distance
        d = (pos**2).sum(axis=1)**.5

        # filters those which are too close or too far
        ind_visible = (pos[:, 2] > self.opt.d_min) * (d < self.opt.d_max)
        N_visible = int(np.sum(ind_visible))

        # self.state = [X, Y, size, R, G, B, alpha]
        self.state = np.ones((N_visible, 7))
        for i in range(2):
            self.state[:, i] = self.opt.mag * pos[ind_visible, i] / pos[ind_visible, 2]
        # size
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
figsize = (16, 16)
ratio = figsize[0]/figsize[1]

if opt.realistic:
    # https://laurentperrinet.github.io/sciblog/posts/2021-03-27-density-of-stars-on-the-surface-of-the-sky.html

    N_X, N_Y = figsize[0], figsize[1]
    N_X, N_Y = int(N_X*opt.dpi), int(N_Y*opt.dpi)
 
    def star(N_X, N_Y, x_pos, y_pos, size_airy, theta, size_airy_ecc, intensity,
                    gamma, model):
        """
        Define the image of a star as a kernel:

        - x_pos, y_pos : position of the center of the blob
        - size_airy_min : axis of minimal variance
        - size_airy_max : axis of maximal variance
        - theta : angle of both angle relative to horizontal ( along Y axis)
        - intensity : relative brightness

        The profile is well approximated by :
        - an Airy disk: https://en.wikipedia.org/wiki/Airy_disk
        - a gaussian https://en.wikipedia.org/wiki/Airy_disk#Approximation_using_a_Gaussian_profile
        - a MOFFAT function https://en.wikipedia.org/wiki/Moffat_distribution

        """
        #X, Y = fx.squeeze(), fy.squeeze()
        X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y))
        X, Y = X.T, Y.T

        # https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
        a = np.cos(theta)**2/(2*size_airy**2) + np.sin(theta)**2/(2*size_airy**2*size_airy_ecc**2)
        b = np.sin(2*theta) * (-1/(4*size_airy**2) + 1/(4*size_airy**2*size_airy_ecc**2))
        c = np.sin(theta)**2/(2*size_airy**2) + np.cos(theta)**2/(2*size_airy**2*size_airy_ecc**2)
        R2 = a * (X-x_pos)**2 + 2 * b * (X-x_pos)*(Y-y_pos) + c * (Y-y_pos)**2

        if model=='airy':
            from scipy.special import jv #(v, z)
            R = np.sqrt(R2)
            image = (jv(1, R) / (R+1e-6))**2
            image /= image.max()
        elif model=='moffat':
            # see https://en.wikipedia.org/wiki/Astronomical_seeing
            beta = .85
            image = (1 + R2)**(-beta)
        else:
            image = np.exp( - R2 )

        #image = sensor_gamma(image, gamma)

        image *= intensity

        return image


    def random_cloud(envelope, events):
        (N_X, N_Y) = envelope.shape
        F_events = np.fft.fftn(events)
        #F_events = np.fft.fftshift(F_events)

        Fz = F_events * envelope
        # de-centering the spectrum
        #Fz = np.fft.ifftshift(Fz)
        #Fz[0, 0, 0] = 0. # removing the DC component
        z = np.fft.ifftn(Fz).real
        return z
    
    def star_env(N_X, N_Y):
        x_star = star(N_X, N_Y, x_pos=N_X//2, y_pos=N_Y//2, 
                      size_airy=1.5, theta=0, size_airy_ecc=1., intensity=.2,
                    gamma=1., model='moffat')
        F_star = np.fft.fftn(x_star)
        #F_star = np.fft.fftshift(F_star)
        return F_star
    F_star = star_env(N_X, N_Y)    

    def model(envelope, events, saturation=1., verbose=False):
        if verbose: print('envelope.shape = ', envelope.shape)
        if verbose: print('events.shape = ', events.shape)
        N_X, N_Y = envelope.shape[0], envelope.shape[1]
        x = random_cloud(envelope, events=events)
        #x = x.reshape((N_X, N_Y))
        if verbose: print(f'{x.min()=:.3f}, {np.median(x)=:.3f}, {x.mean()=:.3f}, {x.max()=:.3f}')
        if saturation < np.inf:
            x = np.minimum(x, saturation)
            if verbose: print(f'{x.min()=:.3f}, {np.median(x)=:.3f}, {x.mean()=:.3f}, {x.max()=:.3f}')
        if verbose: print('x.shape=', x.shape)
        return x

#------------------------------------------------------------
# set up figure and animation
fig, ax = plt.subplots(facecolor=opt.facecolor, #subplot_kw=dict(autoscale_on=False), 
                       figsize=figsize)
fig.set_facecolor(opt.facecolor)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

def animate(i):
    """perform animation step"""
    global box, dt, ax, fig
    box.step(dt)
    ax.cla()
    if opt.realistic:
        # https://laurentperrinet.github.io/sciblog/posts/2021-03-27-density-of-stars-on-the-surface-of-the-sky.html
        a_x = box.state[:, 0].copy()
        a_y = box.state[:, 1].copy()
        #print('0>', a_x.mean(), a_y.mean(), a_x.std(), a_y.std())
        #print('1>', a_x.min(), a_y.min(), a_x.max(), a_y.max())
        #a_x /= opt.mag * ratio * 2  # in (-1, 1)
        #a_y /= opt.mag * ratio * 2  # in (-1, 1)
        #print('1>', a_x.mean(), a_y.mean(), a_x.std(), a_y.std())
        #print('2>', a_x.min(), a_y.min(), a_x.max(), a_y.max())
        a_x += .5   # in (0, 1)
        a_y += .5   # in (0, 1)
        #print('2>', a_x.mean(), a_y.mean(), a_x.std(), a_y.std())
        #print('2>', a_x.min(), a_y.min(), a_x.max(), a_y.max())
        a_x *= N_X/2 # np.max((N_X, N_Y))/2  # in (0, N)
        a_y *= N_X/2 #np.max((N_X, N_Y))/2  # in (0, N)
        #a_x += N_X/2 # np.max((N_X, N_Y))/2  # in (0, N)
        #a_y += N_X/2 #np.max((N_X, N_Y))/2  # in (0, N)
        #print('3>', a_x.min(), a_y.min(), a_x.max(), a_y.max())
        a_x = a_x.astype(int)
        a_y = a_y.astype(int)
        #print('4>', a_x.min(), a_y.min(), a_x.max(), a_y.max())
        valid_idx = (a_x < N_X) * (a_x >= 0) * (a_y < N_Y) * (a_y >= 0)
        #print('5>', valid_idx.sum())
        #print(a_x, mask)
        # size = 1 / d
        # lum = 1 / d^2 = size^2 
        lum = box.state[:, 2]**2 / opt.size 
        
        events = np.zeros((N_X, N_Y))
        events[a_x[valid_idx], a_y[valid_idx]] = lum[valid_idx]
        #for x, y, l in zip(a_x, a_y, lum):
        #    if (x < N_X) and (x >= 0) and (y < N_Y) and (y >= 0):
        #        events[int(x), int(y)] = l
            
        saturation = 1
        x = model(F_star, events, saturation=saturation, verbose=False)
        #x = np.roll(np.roll(x, N_Y//2, 1), N_X//2, 0)
        x = np.roll(np.roll(x, -N_Y//2, 1), -N_X//2, 0) # HACK
        #x = np.roll(np.roll(x, N_Y, 1), N_X, 0)
        ax.imshow(x.T, cmap=plt.gray(), vmin=x.min(), vmax=1);
        #ax.set_xlim(0, N_X) # bounds of the figure
        #ax.set_ylim(0, N_Y)
    else:
        # note: s is the marker size in points**2.
        particles = ax.scatter(box.state[:, 0], box.state[:, 1], marker=opt.marker, c=box.state[:, 3:], s=box.state[:, 2]**2, zorder=1)
        if box.opt.radius > 0:
            circle = plt.Circle((0,0), box.opt.radius, color='k')
            ax.add_artist(circle)
        fixation = ax.scatter([0], [0], marker='+', c='white', s=box.opt.fix_size, zorder=2)
    
        ax.set_xlim(-ratio, ratio) # bounds of the figure
        ax.set_ylim(-1, 1)
    ax.axis('off')
    if box.time_elapsed > opt.T: sys.exit()
    return ax

if opt.vext == 'mp4':
    ani = animation.FuncAnimation(fig, animate, frames=int(opt.T*opt.fps), interval=1000/opt.fps)
    if not opt.fname is None:
        ani.save(opt.fname + '.mp4', fps=opt.fps, extra_args=['-vcodec', 'libx264'], savefig_kwargs=dict(facecolor=opt.facecolor), dpi=opt.dpi)
        # import os
        # os.system('ffmpeg -i starfield.mp4  starfield.gif')

elif opt.vext == 'png':
    import pathlib
    root = pathlib.Path(opt.fname)
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)

    for i_frame in range(N_frame := int(opt.T*opt.fps)):
        print('i_frame =', i_frame, '/', N_frame)
        ax = animate(i_frame)
        fname = root.joinpath(f'frame_{i_frame:06d}.png')
        fig.savefig(fname, dpi=opt.dpi, facecolor=opt.facecolor)

plt.show()
