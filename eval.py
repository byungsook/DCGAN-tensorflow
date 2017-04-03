from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
import time
# try:
#     import moviepy.editor as mpy
# except:
#     from subprocess import call
#     call(['pip', 'install', 'moviepy'])
#     import imageio
#     imageio.plugins.ffmpeg.download()
import imageio

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import tensorflow as tf

import model



flags = tf.app.flags
flags.DEFINE_string('log_dir', '../eval/test',
                    """Directory where to write event logs and checkpoint. [*]""")
flags.DEFINE_string('checkpoint_dir', '../log/test',
                    """Directory of checkpoint. [*]""")
flags.DEFINE_integer('batch_size', 100,
                     """The size of batch images [64]""")
flags.DEFINE_integer('image_size', 128,
                     """The size of image to use. [128]""")
flags.DEFINE_integer('image_depth', 2,
                     """The depth of image to use. [2]""")
flags.DEFINE_integer('z_dim', 3,
                     """Dimension of z. [100]""")
flags.DEFINE_string('mp4_file_name', 'vel_%d.mp4',
                    """mp4 file name. [*]""")
flags.DEFINE_string('mp4_rev_file_name', 'vel_rev_%d.mp4',
                    """reverse mp4 file name. [*]""")
flags.DEFINE_float('duration', 5.0,
                    """mp4 duration. [5]""")
FLAGS = flags.FLAGS


def eval():
    ### print flags
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    ### build a network
    image_dims = [FLAGS.image_size, FLAGS.image_size, FLAGS.image_depth]
    z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim], name='z')
    x_gen = model.generator(z)

    # z_sample = np.ones(shape=[FLAGS.z_dim, FLAGS.batch_size, FLAGS.z_dim]) * -1
    # z_sample = np.zeros(shape=[FLAGS.z_dim, FLAGS.batch_size, FLAGS.z_dim])
    z_sample = np.ones(shape=[FLAGS.z_dim, FLAGS.batch_size, FLAGS.z_dim])
    z_sample[0,:,0] = np.linspace(-1, 1, FLAGS.batch_size)
    z_sample[1,:,1] = np.linspace(-1, 1, FLAGS.batch_size)
    z_sample[2,:,2] = np.linspace(-1, 1, FLAGS.batch_size)

    ### Start running operations on the Graph.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf.Session(config=config)
    
    # Create a saver (restorer).
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    assert(ckpt and FLAGS.checkpoint_dir)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
    print('%s: Pre-trained model restored from %s' % (datetime.now(), ckpt_name))
    
    X, Y = np.meshgrid(np.arange(0, FLAGS.image_size), np.arange(0, FLAGS.image_size))

    # Start to eval
    print('%s: start to eval' % datetime.now())
    for dim in xrange(FLAGS.z_dim):
        x_generated = sess.run(x_gen, feed_dict={z: z_sample[dim]})

        mp4_path = os.path.join(FLAGS.log_dir, FLAGS.mp4_file_name % dim)
        fps = FLAGS.batch_size / FLAGS.duration
        mp4_writer = imageio.get_writer(mp4_path, fps=fps)

        for i in xrange(FLAGS.batch_size):
            U = x_generated[i,:,:,0]
            V = x_generated[i,:,:,1]
            M = np.hypot(U, V)
            
            fig = plt.figure()
            fig.set_size_inches(1, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            plt.axis('equal')
            plt.xlim([0,FLAGS.image_size])
            plt.ylim([0,FLAGS.image_size])
            ax.quiver(X, Y, U, V, M) # , width=0.001, headwidth=1)

            quiver_path = os.path.join(FLAGS.log_dir, 'quiver_d%d_%d.png' % (dim, i))
            plt.savefig(quiver_path, dpi=FLAGS.image_size*4)
            plt.close(fig)
            print('save', quiver_path)

            im = imageio.imread(quiver_path)
            mp4_writer.append_data(im)
            # break
        mp4_writer.close()

        mp4_rev_path = os.path.join(FLAGS.log_dir, FLAGS.mp4_rev_file_name % dim)
        mp4_writer = imageio.get_writer(mp4_rev_path, fps=fps)
        for i in reversed(xrange(FLAGS.batch_size)):
            quiver_path = os.path.join(FLAGS.log_dir, 'quiver_d%d_%d.png' % (dim, i))
            im = imageio.imread(quiver_path)
            mp4_writer.append_data(im)
        mp4_writer.close()


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('code'):
        working_path = os.path.join(current_path, 'fluid_feature/code')
        os.chdir(working_path)

    # create log directory
    if FLAGS.log_dir.endswith('log'):
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, datetime.now().isoformat().replace(':', '-'))
    elif tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    eval()


if __name__ == '__main__':
    tf.app.run()
