from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


flags = tf.app.flags
flags.DEFINE_integer('f_dim', 64, # 64, 128, 256, 512 -> 1
                     """Dimension of filters in first conv layer [64]""")
FLAGS = flags.FLAGS


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def _conv2d(input_, output_dim, k=5, s=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def _deconv2d(input_, output_shape, k=5, s=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k, k, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # try:
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s, s, 1])
        # # Support for verisons of TensorFlow before 0.7.0
        # except AttributeError:
        # deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
         
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv

     
def _lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)


def _linear(input_, output_size, name='linear', stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable('matrix', [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def discriminator(x, reuse=False):
    '''
    x -> True or False
    '''
    with tf.variable_scope('discriminator') as scope:
        if reuse: scope.reuse_variables()

        d_bn1 = BatchNorm(name='d_bn1')
        d_bn2 = BatchNorm(name='d_bn2')
        d_bn3 = BatchNorm(name='d_bn3')
        
        h0 = _lrelu(_conv2d(x, FLAGS.f_dim, name='d_h0_conv')) # 128x128x64
        h1 = _lrelu(d_bn1(_conv2d(h0, FLAGS.f_dim*2, name='d_h1_conv'))) # 64x64x128
        h2 = _lrelu(d_bn2(_conv2d(h1, FLAGS.f_dim*4, name='d_h2_conv'))) # 32x32x256
        h3 = _lrelu(d_bn3(_conv2d(h2, FLAGS.f_dim*8, name='d_h3_conv'))) # 16x16x512
        h4 = _linear(tf.reshape(h3, [FLAGS.batch_size, -1]), 1, name='d_h3_lin') # 1x1
    return h4, tf.nn.sigmoid(h4)


def generator(z, sample=False):
    '''
    z -> x_fake
    '''
    with tf.variable_scope('generator') as scope:
        train = True
        if sample:  
            scope.reuse_variables()
            train = False

        g_bn0 = BatchNorm(name='g_bn0')
        g_bn1 = BatchNorm(name='g_bn1')
        g_bn2 = BatchNorm(name='g_bn2')
        g_bn3 = BatchNorm(name='g_bn3')

        s = int(FLAGS.image_size / 16.0)
        o = [FLAGS.batch_size, s, s, FLAGS.f_dim*8]
        
        # project `z` and reshape
        z_ = _linear(z, s*s*FLAGS.f_dim*8, name='g_h0_lin')
        h0 = tf.nn.relu(g_bn0(tf.reshape(z_, shape=o), train=train)) # 8x8x512

        o = [FLAGS.batch_size, s*2, s*2, FLAGS.f_dim*4] # 16x16x256
        h1 = tf.nn.relu(g_bn1(_deconv2d(h0, o, name='g_h1'), train=train))

        o = [FLAGS.batch_size, s*4, s*4, FLAGS.f_dim*2] # 32x32x128
        h2 = tf.nn.relu(g_bn2(_deconv2d(h1, o, name='g_h2'), train=train))

        o = [FLAGS.batch_size, s*8, s*8, FLAGS.f_dim] # 64x64x64
        h3 = tf.nn.relu(g_bn3(_deconv2d(h2, o, name='g_h3'), train=train))

        o = [FLAGS.batch_size, s*16, s*16, FLAGS.image_depth] # 128x128x2
        h4 = _deconv2d(h3, o, name='g_h4')
    return tf.nn.tanh(h4)


def build_model(x, z):
    # discriminator
    y_logits, y = discriminator(x)
    
    # generator
    x_fake = generator(z)
    y_fake_logits, y_fake = discriminator(x_fake, reuse=True)
    
    show_all_variables()
    return y_logits, y, y_fake_logits, y_fake
    

def loss(y, y_fake):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=tf.ones_like(y)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_fake, labels=tf.zeros_like(y)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_fake, labels=tf.ones_like(y)))
    return d_loss_real, d_loss_fake, g_loss