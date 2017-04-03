from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import random
import multiprocessing.managers
import multiprocessing.pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64,
                     """The size of batch images [64]""")
flags.DEFINE_integer('image_size', 128,
                     """The size of image to use. [128]""")
flags.DEFINE_integer('image_depth', 2,
                     """The depth of image to use. [2]""")
flags.DEFINE_integer('z_dim', 3,
                     """Dimension of z. [100]""")
flags.DEFINE_string('data_dir', "../data/velocity",
                    """Directory of data set [*]""")
flags.DEFINE_float('num_processors', multiprocessing.cpu_count(),
                   """maximum magnitude of velocity. [15]""")
flags.DEFINE_float('v_max', 15,
                   """maximum magnitude of velocity. [15]""")
FLAGS = flags.FLAGS


class MPManager(multiprocessing.managers.SyncManager):
    pass
MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)


class Param(object):
    def __init__(self):
        self.data_dir = FLAGS.data_dir
        self.v_max = FLAGS.v_max


class BatchManager(object):
    def __init__(self):
        for _, _, self._data_list in os.walk(FLAGS.data_dir):
            break

        self._data_list.sort()
        
        # read data generation arguments
        self._args = {}
        with open(os.path.join(FLAGS.data_dir, self._data_list.pop(0)), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self._args[arg] = arg_value

        FLAGS.image_height = int(self._args['resolution'])
        FLAGS.image_width = FLAGS.image_height
        
        # read max velocity magnitude
        with open(os.path.join(FLAGS.data_dir, self._data_list.pop()), 'r') as f:
            FLAGS.v_max = float(f.read())
        
        self.num_examples_per_epoch = len(self._data_list)
        self.num_epoch = 0
        self._file_id = 0

        image_dim = [FLAGS.image_size, FLAGS.image_size, FLAGS.image_depth]

        if FLAGS.num_processors > FLAGS.batch_size:
            FLAGS.num_processors = FLAGS.batch_size

        if FLAGS.num_processors == 1:
            self.x_batch = np.zeros([FLAGS.batch_size]+image_dim, dtype=np.float)
        else:
            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)
            
            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size]+image_dim, dtype=np.float)
            self._file_path_batch = self._mpmanager.list(['' for _ in xrange(FLAGS.batch_size)])
            self._func = partial(train_set, file_path_batch=self._file_path_batch, x_batch=self.x_batch, FLAGS=Param())


    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            file_path_batch = [None] * FLAGS.batch_size
            for i in xrange(FLAGS.batch_size):
                if self._file_id % self.num_examples_per_epoch == 0:
                    self.num_epoch += 1
                    self._file_id = 0
                    random.shuffle(self._data_list)

                file_path_batch[i] = self._data_list[self._file_id]
                train_set(i, file_path_batch, self.x_batch, FLAGS)
                self._file_id += 1
                
        else:
            for i in xrange(FLAGS.batch_size):
                if self._file_id % self.num_examples_per_epoch == 0:
                    self.num_epoch += 1
                    self._file_id = 0
                    random.shuffle(self._data_list)

                self._file_path_batch[i] = self._data_list[self._file_id]
                self._file_id += 1
        
            self._pool.map(self._func, range(FLAGS.batch_size))

        z_batch = np.random.uniform(-1, 1, size=[FLAGS.batch_size, FLAGS.z_dim])
        return self.x_batch, z_batch


def train_set(batch_id, file_path_batch, x_batch, FLAGS):
    UV = np.loadtxt(os.path.join(FLAGS.data_dir, file_path_batch[batch_id]))
    UV /= FLAGS.v_max # normalize
    U = UV[:,::2]
    V = UV[:,1::2]

    x_batch[batch_id,:,:,0] = U
    x_batch[batch_id,:,:,1] = V

    # # for debug

    # plt.figure()
    # plt.subplot(121)
    # plt.axis('equal')
    # plt.xlim([0,FLAGS.image_size])
    # plt.ylim([0,FLAGS.image_size])
    # plt.imshow(U, origin='lower', cmap='bwr', vmin=-1, vmax=1)

    # plt.subplot(122)
    # plt.axis('equal')
    # plt.xlim([0,FLAGS.image_size])
    # plt.ylim([0,FLAGS.image_size])
    # plt.imshow(V, origin='lower', cmap='bwr', vmin=-1, vmax=1)
    # plt.show()


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('code'):
        working_path = os.path.join(current_path, 'fluid_feature/code')
        os.chdir(working_path)
    
    FLAGS.num_processors = 8
    FLAGS.batch_size = 8

    batch_manager = BatchManager()
    x_batch, z_batch = batch_manager.batch()

    image_hw = [FLAGS.image_size, FLAGS.image_size]
    plt.figure()
    for i in xrange(FLAGS.batch_size):
        plt.subplot(121)
        plt.axis('equal')
        plt.xlim([0,FLAGS.image_size])
        plt.ylim([0,FLAGS.image_size])
        plt.imshow(np.reshape(x_batch[i,:,:,0], image_hw), origin='lower', cmap='bwr', vmin=-1, vmax=1)

        plt.subplot(122)
        plt.axis('equal')
        plt.xlim([0,FLAGS.image_size])
        plt.ylim([0,FLAGS.image_size])
        plt.imshow(np.reshape(x_batch[i,:,:,1], image_hw), origin='lower', cmap='bwr', vmin=-1, vmax=1)
        plt.show()

    print('done')