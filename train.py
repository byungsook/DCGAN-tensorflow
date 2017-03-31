from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
import time

import numpy as np
import tensorflow as tf

import model
import batch_data



flags = tf.app.flags
flags.DEFINE_string('log_dir', '../log/test', 
                    """Directory where to write event logs and checkpoint. [*]""")
flags.DEFINE_string('checkpoint_dir', '',
                    """Directory of checkpoint. [*]""")
flags.DEFINE_integer('num_epochs', 30, 
                     """Epoch to train [30]""")
flags.DEFINE_float('learning_rate', 0.0002,
                   """Learning rate of for adam [0.0002]""")
flags.DEFINE_float('beta1', 0.5,
                   """Momentum term of adam [0.5]""")
flags.DEFINE_integer('train_size', np.inf,
                     """The size of train images [np.inf]""")
flags.DEFINE_integer('save_steps', 100,
                     """save steps""")
flags.DEFINE_integer('max_images', 64,
                     """max # images to save.""")
FLAGS = flags.FLAGS


def train():
    ### print flags
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    ### build a network
    image_dims = [FLAGS.image_size, FLAGS.image_size, FLAGS.image_depth]
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size]+image_dims, name='x')
    z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size,FLAGS.z_dim], name='z')
    y_logits, y, y_fake_logits, y_fake = model.build_model(x, z)

    ### losses
    d_loss_real, d_loss_fake, g_loss = model.loss(y_logits, y_fake_logits)
    d_loss = d_loss_real + d_loss_fake

    ### vars
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    ### optimizers
    d_opt = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                   beta1=FLAGS.beta1).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                   beta1=FLAGS.beta1).minimize(g_loss, var_list=g_vars)

    ### summary
    y_summary = tf.summary.histogram('d_real', y)
    y_fake_summary = tf.summary.histogram('d_fake', y_fake)

    d_loss_summary = tf.summary.scalar('d_loss', d_loss)
    d_loss_real_summary = tf.summary.scalar('d_loss_real', d_loss_real)
    d_loss_fake_summary = tf.summary.scalar('d_loss_fake', d_loss_fake)
    
    g_loss_summary = tf.summary.scalar('g_loss', g_loss)
    
    d_summary = tf.summary.merge([y_summary, d_loss_summary, d_loss_real_summary, d_loss_fake_summary])
    g_summary = tf.summary.merge([y_fake_summary, g_loss_summary])

    ### Start running operations on the Graph.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf.Session(config=config)
    
    # Create a saver (restorer).
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and FLAGS.checkpoint_dir:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
        print('%s: Pre-trained model restored from %s' % 
              (datetime.now(), ckpt_name))
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)


    ####################################################################
    ### create batch manager
    batch_manager = batch_data.BatchManager()

    ### create sampler for testing
    sample = model.generator(z, sample=True)
    x_sample, z_sample = batch_manager.batch()
    sample_dim = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1]
    sample_x = tf.placeholder(dtype=tf.float32, shape=sample_dim)
    sample_y = tf.placeholder(dtype=tf.float32, shape=sample_dim)
    sample_x_summary = tf.summary.image('sample_x', sample_x, max_outputs=FLAGS.max_images)
    sample_y_summary = tf.summary.image('sample_y', sample_y, max_outputs=FLAGS.max_images)
    sample_x_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,'sample_x'), sess.graph)
    sample_y_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,'sample_y'), sess.graph)

    # Start to train.
    print('%s: start to train' % datetime.now())
    end_step = int(np.ceil(batch_manager.num_examples_per_epoch / float(FLAGS.batch_size) * FLAGS.num_epochs))
    for step in xrange(end_step):
        # Train one step.
        start_time = time.time()
        x_batch, z_batch = batch_manager.batch()

        sess.run(d_opt, feed_dict={x: x_batch, z: z_batch})
        sess.run(g_opt, feed_dict={z: z_batch})
        sess.run(g_opt, feed_dict={z: z_batch})
        duration = time.time() - start_time

        d_loss_real_val, d_loss_fake_val, g_loss_val = sess.run([d_loss_real, d_loss_fake, g_loss],
                                                                feed_dict={x: x_batch, z: z_batch})
        print('%s: [epoch %d][step %d/%d] d_r = %.2f, d_f = %.2f, g = %.2f (%.3f sec/batch)' % 
              (datetime.now(), batch_manager.num_epoch, step, end_step,
               d_loss_real_val, d_loss_fake_val, g_loss_val, duration))

        # write summary
        d_summary_str, g_summary_str = sess.run([d_summary, g_summary],
                                                feed_dict={x: x_batch, z: z_batch})
        summary_writer.add_summary(d_summary_str, step)
        summary_writer.add_summary(g_summary_str, step)

        # Save the model checkpoint and test with sampler periodically.
        if step % FLAGS.save_steps == 0 or (step + 1) == end_step:
            checkpoint_path = os.path.join(FLAGS.log_dir, 'velocity.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

            samples, d_loss_val, g_loss_val = sess.run([sample, d_loss, g_loss],
                                                       feed_dict={x: x_sample, z: z_sample})
            print('%s: [epoch %d][step %d/%d][sample] d = %.2f, g = %.2f' % 
                  (datetime.now(), batch_manager.num_epoch, step, end_step, d_loss_val, g_loss_val))

            x_str, y_str = sess.run([sample_x_summary, sample_y_summary],
                feed_dict={sample_x: np.reshape(samples[:,:,:,0], newshape=sample_dim),
                           sample_y: np.reshape(samples[:,:,:,1], newshape=sample_dim)})
            
            x_summary_tmp = tf.Summary()
            y_summary_tmp = tf.Summary()
            x_summary_tmp.ParseFromString(x_str)
            y_summary_tmp.ParseFromString(y_str)
            for i in xrange(FLAGS.max_images):
                x_summary_tmp.value[i].tag = '%06d/%02d' % (step, i)
                y_summary_tmp.value[i].tag = '%06d/%02d' % (step, i)
                sample_x_writer.add_summary(x_summary_tmp, step)
                sample_y_writer.add_summary(y_summary_tmp, step)


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

    train()


if __name__ == '__main__':
    tf.app.run()
