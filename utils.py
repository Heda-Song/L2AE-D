import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False, pooling=True, drop=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]
    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope, drop)
    if pooling:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope, drop):
    if FLAGS.K_shot == 1:
        scale_flag = True
    else:
        scale_flag = False
    normed = tf_layers.batch_norm(inp, scale=scale_flag, activation_fn=activation, reuse=reuse, scope=scope)
    if drop:
        if '1' in scope:
            normed = tf.nn.dropout(normed, 0.5, seed=1)
        elif '2' in scope:
            normed = tf.nn.dropout(normed, 0.5, seed=12)
        elif '4' in scope:
            normed = tf.nn.dropout(normed, 0.5, seed=1234)
        elif '5' in scope:
            normed = tf.nn.dropout(normed, 0.5, seed=12345)
    return normed

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.K_shot
