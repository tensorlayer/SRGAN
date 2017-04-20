#! /usr/bin/python
# -*- coding: utf8 -*-

# import dial_nn
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

def super_resolution_64_256(t_image, is_train=False, reuse=False):
    """ input 64x64x3, output 256x256x3""" # Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("super_resolution_64_256", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c1_%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='b1_%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c2_%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='b2_%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'add1_%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c3')
        n = BatchNormLayer(nn, is_train=is_train, gamma_init=gamma_init, name='b3')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c4')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='subpixel1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c5')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='subpixel2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

def super_resolution_256_discriminator(t_image, is_train=False, reuse=False):
    """ input 256x256x3, output real/fake """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("super_resolution_256_discriminator", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=gamma_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)
        return n, logits

def vgg16_cnn_emb(t_image, reuse=False):
    """ t_image = 244x244 [0~255] """
    with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        t_image = (t_image + 1) * 127.5 # Hao DH if input is [-1, 1]

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in = InputLayer(t_image - mean, name='vgg_input_im')
        """ conv1 """
        network = tl.layers.Conv2dLayer(net_in,
                        act = tf.nn.relu,
                        shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv1_1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv1_2')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='vgg_pool1')
        """ conv2 """
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv2_1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv2_2')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='vgg_pool2')
        """ conv3 """
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv3_1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv3_2')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv3_3')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='vgg_pool3')
        """ conv4 """
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv4_1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv4_2')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv4_3')

        network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='vgg_pool4')
        conv4 = network

        """ conv5 """
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv5_1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv5_2')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name ='vgg_conv5_3')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='vgg_pool5')

        network = FlattenLayer(network, name='vgg_flatten')

        # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
        # new_network = tl.layers.DenseLayer(network, n_units=4096,
        #                     act = tf.nn.relu,
        #                     name = 'vgg_out/dense')
        #
        # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
        # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
        #             b_init=None, name='vgg_out/out')
        return conv4, network
