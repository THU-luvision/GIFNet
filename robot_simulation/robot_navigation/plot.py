import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib
from config import *
import os.path as osp
import os
from contextlib import contextmanager
import matplotlib.animation as mpl_animation


def vis_fig(trajs, savedir, flag=False):
    '''
    visulaize the trajectories of each person
    '''
    os.makedirs(savedir, exist_ok=True)

    color1 = (1, 0, 0)
    color2 = (0, 1, 0)
    color3 = (0, 0, 1)
    colors = [color1, color2, color3]

    for fid in range(len(trajs[0])):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set(xlim=(width[0], width[1]), ylim=(height[0], height[1]))
        # ax.set_aspect('equal')
        # plt.grid()
        # ax.grid()

        for i, traj in enumerate(trajs):
            # circle1 = plt.Circle((traj[fid, 0], traj[fid, 1]), rad_sin, color=colors[i], zorder=0.1)
            circle2 = plt.Circle((traj[fid, 0], traj[fid, 1]), 0.2, color=colors[(i + 1) % 2], zorder=0.2)

            # draw the decision field
            if fid != len(trajs[0]) - 1 and view_ang and flag:
                # print('yyyyyy')
                cur_vel = traj[fid + 1] - traj[fid]
                vel_ang = np.arctan2(cur_vel[1], cur_vel[0])
                up_bound = np.rad2deg(vel_ang + view_ang)
                low_bound = np.rad2deg(vel_ang - view_ang)
                # print(low_bound, up_bound)
                # Wedge(center,radius,theta1,theta2)
                sector = Wedge((traj[fid, 0], traj[fid, 1]), 0.5, low_bound, up_bound, zorder=0.3)
                ax.add_patch(sector)

            # ax.add_artist(circle1)
            ax.add_artist(circle2)

        # plt.show()
        savename = osp.join(savedir, '{}.jpg'.format(fid))
        fig.savefig(savename, dpi=500)
        print('fid {0:03d} save'.format(fid))
        plt.close(fig)


def compare_trajs(dirs):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set(aspect='auto', xlabel='x [m]', ylabel='y [m]', title='1v1', axisbelow=True)

    datas = []
    models = ['SFM', 'UNL']
    for ped, dir in enumerate(dirs):
        print(ped)
        data = np.load(dir)
        datas.append(data)
        print(data.shape)

    data0 = datas[0]
    window = data0.shape[1]
    for idx, data in enumerate(datas[1:]):
        # diff_x = data[0, :window, 0] - data0[0, :, 0]
        x = [i for i in range(window)]
        diff_y = data[0, :window, 1] - data0[0, :, 1]
        ax.plot(x, diff_y, '-o', label='model {}'.format(models[idx]), markersize=0.5)

        ax.legend()
    fig.savefig('./vis_results/compare/1v1.png', dpi=300)
    plt.show()
    plt.close()


@contextmanager
def animation(n, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    print('ploting..')
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')

    context = {'ax': ax, 'update_function': None}
    yield context

    ani = mpl_animation.FuncAnimation(fig, context['update_function'], range(n))
    if movie_file:
        ani.save(movie_file, writer=writer)
    fig.show()
    plt.close(fig)


@contextmanager
def vis_gif(trajs, savedir):
    with animation(trajs.shape[1], savedir, writer='imagemagick') as context:
        ax = context['ax']
        # yield ax

        actors = []
        for ped in range(trajs.shape[0]):
            p = plt.Circle((trajs[ped, 0, 0], trajs[ped, 0, 1]), radius=0.2, facecolor='black' if trajs[ped, 0, 0] < 115 else 'white', edgecolor='black')
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                p.center = trajs[ped, i, 0:2]

        ax.set(aspect='1.0', xlabel='x [m]', ylabel='y [m]', title='1v1', axisbelow=True)
        ax.set(xlim=(width[0], width[1]), ylim=(height[0], height[1]))

        context['update_function'] = update


def vis_traj(traj,savename):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set(aspect='auto', xlabel='x [m]', ylabel='y [m]', title='1v1',axisbelow=True)

    for idx in range(traj.shape[0]):
        x = traj[idx,:,0]
        y = traj[idx,:,1]
        ax.plot(x, y, '-o', label='actor {}'.format(idx), markersize=0.5)

    ax.legend()
    fig.savefig('./vis_results/compare/{}.png'.format(savename), dpi=300)
    #plt.show()
    plt.close()
def vis_vel(vels,savename):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set(aspect='auto', xlabel='t', ylabel='vel', title='1v1',axisbelow=True)
    for idx in range(vels.shape[0]):
        y = np.linalg.norm(vels[idx],axis=1)

        x = np.linspace(0, vels[idx].shape[0],vels[idx].shape[0])
        ax.plot(x, y, '-o', label='actor {}'.format(idx), markersize=0.5)

    ax.legend()
    fig.savefig('./vis_results/vel/{}.png'.format(savename), dpi=300)
    #plt.show()
    plt.close()
def vis_deltav(vels,savename):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set(aspect='auto', xlabel='t', ylabel='delta_v', title='1v1',axisbelow=True)
    for idx in range(vels.shape[0]):
        
        v_next = np.zeros_like(vels[idx])
        v_next[0:-1] = vels[idx,1:,] 
        delta_v = v_next - vels[idx]
        y = np.linalg.norm(delta_v,axis=1)

        x = np.linspace(0, vels[idx].shape[0],vels[idx].shape[0])
        ax.plot(x, y, '-o', label='actor {}'.format(idx), markersize=0.5)

    ax.legend()
    fig.savefig('./vis_results/delta_v/{}.png'.format(savename), dpi=300)
    #plt.show()
    plt.close()    
if __name__ == '__main__':
    name = 'single-1v1'
    data = np.load('./npys/{}.npy'.format(name))
    vis_traj(data,name )
