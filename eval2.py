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
import scipy.misc

from matplotlib.transforms import Bbox
import tensorflow as tf

import model



flags = tf.app.flags
flags.DEFINE_string('log_dir', '../eval/z-space/z3',
                    """Directory where to write event logs and checkpoint. [*]""")
flags.DEFINE_string('checkpoint_dir', '../log/train',
                    """Directory of checkpoint. [*]""")
flags.DEFINE_integer('batch_size', 100,
                     """The size of batch images [64]""")
flags.DEFINE_integer('image_size', 128,
                     """The size of image to use. [128]""")
flags.DEFINE_integer('image_depth', 2,
                     """The depth of image to use. [2]""")
flags.DEFINE_integer('z_dim', 3,
                     """Dimension of z. [100]""")
flags.DEFINE_string('img_file_name', '%s_%03d_%.3f.png',
                    """image file name. [*]""")
flags.DEFINE_string('mp4_file_name', '%s.mp4',
                    """mp4 file name. [*]""")
flags.DEFINE_float('duration', 5.0,
                    """mp4 duration. [5]""")
FLAGS = flags.FLAGS
X, Y = np.meshgrid(np.arange(0, FLAGS.image_size), np.arange(0, FLAGS.image_size))


def merge(imgs, imgs_per_line):
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    merged_img = np.zeros((h*imgs_per_line, w*imgs_per_line, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = imgs_per_line - 1 - (idx // imgs_per_line)
        c = idx % imgs_per_line
        merged_img[r*h:r*h+h, c*w:c*w+w, :] = img[:,:,:3]
    return merged_img


def draw_images(x_generated, sqr_bsize):
    img_size = 256
    imgs = []

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

        quiver_path = os.path.join(FLAGS.log_dir, 'tmp/quiver_%d.png' % i)
        plt.savefig(quiver_path, dpi=img_size)
        plt.close(fig)
        # print('save', quiver_path)

        img = scipy.misc.imread(quiver_path)
        imgs.append(img)
        os.remove(quiver_path)
        # plt.imshow(img)
        # plt.show()
        # break

    img = merge(imgs, sqr_bsize)
    # plt.imshow(img)
    # plt.show()

    return img


def eval():
    ### print flags
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    ### build a network
    image_dims = [FLAGS.image_size, FLAGS.image_size, FLAGS.image_depth]
    z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim], name='z')
    x_gen = model.generator(z)

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
    
    sqr_bsize = int(FLAGS.batch_size ** 0.5) # 10
    assert(float(sqr_bsize) == FLAGS.batch_size ** 0.5)
    
    for axis in xrange(3):
        z_str = 'z%d' % (axis+1)

        linspc = np.linspace(-1, 1, sqr_bsize)    
        z_sample = np.ones(shape=[FLAGS.batch_size, FLAGS.z_dim]) * -1    
        for i in xrange(sqr_bsize):
            z_val = linspc[i]
            for j in xrange(sqr_bsize):
                batch_id = i*10 + j
                z_sample[batch_id,(axis+1)%3] = z_val # row
                z_sample[batch_id,(axis+2)%3] = linspc[j] # col


        linspc = np.linspace(-1, 1, FLAGS.batch_size)

        mp4_path = os.path.join(FLAGS.log_dir, FLAGS.mp4_file_name % z_str)
        fps = FLAGS.batch_size / FLAGS.duration
        mp4_writer = imageio.get_writer(mp4_path, fps=fps)

        # Start to eval
        print('%s: start to eval' % datetime.now())
        for i in xrange(FLAGS.batch_size):
            z_val = linspc[i]
            z_sample[:,axis] = z_val
            x_generated = sess.run(x_gen, feed_dict={z: z_sample})

            img = draw_images(x_generated, sqr_bsize)
            img_path =  os.path.join(FLAGS.log_dir, FLAGS.img_file_name % (z_str, i, z_val))
            scipy.misc.imsave(img_path, img)

            im = imageio.imread(img_path)
            mp4_writer.append_data(im)
            print('save', img_path)

        mp4_writer.close()
        
    print('done')


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
    tf.gfile.MakeDirs(FLAGS.log_dir + '/tmp')

    eval()


if __name__ == '__main__':
    tf.app.run()
