import argparse
import sys
from datetime import datetime
import time
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='../data/velocity')
parser.add_argument("--path_format", type=str, default='v_%d_%d_%d.txt')
parser.add_argument("--num_src_x_pos", type=int, default=10)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)
parser.add_argument("--num_src_radius", type=int, default=10)
parser.add_argument("--min_src_radius", type=float, default=0.06)
parser.add_argument("--max_src_radius", type=float, default=0.15)
parser.add_argument("--num_frames", type=int, default=100)

parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--gravity", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--time_step", type=float, default=1.0)

args = parser.parse_args()


def read_vel(p1, p2, sim_id, v_max):
    UV = np.loadtxt(os.path.join(args.log_dir, 'v_%d_%d_%d.txt' % (p1, p2, sim_id)))
    UV /= v_max # normalize
    U = UV[:,::2]
    V = UV[:,1::2]
    return U, V


def main(p1, p2):
    print(p1, p2)
    res = args.resolution
    try:
        v_max = np.loadtxt(os.path.join(args.log_dir, 'v_max.txt'))
    except:
        v_max = 15.0

    start = p1 * args.num_src_radius * args.num_frames + p2 * args.num_frames
    end = start + args.num_frames
    step = args.num_frames / 10
    # start = end -1

    X, Y = np.meshgrid(np.arange(0, res), np.arange(0, res))

    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    try:
        mng.resize(*mng.window.maxsize())
        # mng.frame.Maximize(True)
    except:
        mng.window.showMaximized()
        # mng.window.state('zoomed') # windows

    # plt.ion()
    # plt.show()

    im = [None, None, None]
    for sim_id in xrange(start,end,step):
        U, V = read_vel(p1, p2, sim_id, v_max)
        
        plt.subplot(131)
        M = np.hypot(U, V)
        if not im[0]:
            plt.axis('equal')
            plt.xlim([0,args.resolution])
            plt.ylim([0,args.resolution])
            im[0] = plt.quiver(X, Y, U, V, width=0.0005, headwidth=1, scale=v_max)
        else:
            im[0].set_UVC(U, V)

        # plt.subplot(222)
        # plt.axis('equal')
        # if not quiver:
        #     plt.streamplot(X, Y, U, V, color=M, density=3)
        # else:
        #     plt.cla()
        #     plt.streamplot(X, Y, U, V, color=M, density=3)

        plt.subplot(132)
        if not im[1]:
            plt.axis('equal')
            plt.xlim([0,args.resolution])
            plt.ylim([0,args.resolution])
            im[1] = plt.imshow(U, origin='lower', cmap='bwr', vmin=-1, vmax=1)
        else:
            im[1].set_data(U)

        plt.subplot(133)
        if not im[2]:
            plt.axis('equal')
            plt.xlim([0,args.resolution])
            plt.ylim([0,args.resolution])
            im[2] = plt.imshow(V, origin='lower', cmap='bwr', vmin=-1, vmax=1)
        else:
            im[2].set_data(V)

        plt.draw()
        plt.pause(0.00001)

    plt.show()
    print('done')

if __name__ == '__main__':
    # if release mode, change current path
    working_path = os.getcwd()
    if working_path.endswith('dev'):
        working_path = os.path.join(working_path, 'fluid_feature/code')
        os.chdir(working_path)
    else:
        working_path = os.path.join(working_path, '..')
        os.chdir(working_path)

    print(working_path)
    print('p1: pos, p2: src size')
    main(p1=0, p2=4)
    main(p1=4, p2=4)
    main(p1=9, p2=4)    
    
    main(p1=4, p2=0)
    main(p1=4, p2=4)
    main(p1=4, p2=9)

