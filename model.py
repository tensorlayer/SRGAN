#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.models import Model

def get_G(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
        nn = BatchNorm2d(gamma_init=g_init)(nn)
        nn = Elementwise(tf.add)([n, nn])
        n = nn

    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=g_init)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
    G = Model(inputs=nin, outputs=nn, name="generator")
    return G

def get_D(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nin = Input(input_shape)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nin)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=gamma_init)(n)
    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])

    n = Flatten()(n)
    no = Dense(n_units=1, W_init=w_init)(n)
    D = Model(inputs=nin, outputs=no, name="discriminator")
    return D

# def get_G2(input_shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#
#     n = InputLayer(t_image, name='in')
#     n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
#     temp = n
#
#     # B residual blocks
#     for i in range(16):
#         nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
#         nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
#         nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
#         nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
#         nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
#         n = nn
#
#     n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
#     n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
#     n = ElementwiseLayer([n, temp], tf.add, name='add3')
#     # B residual blacks end
#
#     # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
#     # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
#     #
#     # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
#     # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
#
#     ## 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
#     n = UpSampling2dLayer(n, size=[size[1] * 2, size[2] * 2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')
#     n = Conv2d(n, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')  # <-- may need to increase n_filter
#     n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')
#
#     n = UpSampling2dLayer(n, size=[size[1] * 4, size[2] * 4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d')
#     n = Conv2d(n, 32, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')  # <-- may need to increase n_filter
#     n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')
#
#     n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
#     return n


# def SRGAN_d2(t_image, is_train=False, reuse=False):
#     """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
#     feature maps (n) and stride (s) feature maps (n) and stride (s)
#     """
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     b_init = None
#     g_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
#         # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
#         n = InputLayer(t_image, name='in')
#         n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')
#
#         n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')
#
#         n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')
#
#         n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')
#
#         n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')
#
#         n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')
#
#         n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')
#
#         n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')
#
#         n = FlattenLayer(n, name='f')
#         n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
#         n = DenseLayer(n, n_units=1, name='out')
#
#         logits = n.outputs
#         n.outputs = tf.nn.sigmoid(n.outputs)
#
#         return n, logits


# def Vgg19_simple_api(rgb, reuse):
#     """
#     Build the VGG 19 Model
#
#     Parameters
#     -----------
#     rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
#     """
#     VGG_MEAN = [103.939, 116.779, 123.68]
#     with tf.variable_scope("VGG19", reuse=reuse) as vs:
#         start_time = time.time()
#         print("build model started")
#         rgb_scaled = rgb * 255.0
#         # Convert RGB to BGR
#         if tf.__version__ <= '0.11':
#             red, green, blue = tf.split(3, 3, rgb_scaled)
#         else:  # TF 1.0
#             # print(rgb_scaled)
#             red, green, blue = tf.split(rgb_scaled, 3, 3)
#         assert red.get_shape().as_list()[1:] == [224, 224, 1]
#         assert green.get_shape().as_list()[1:] == [224, 224, 1]
#         assert blue.get_shape().as_list()[1:] == [224, 224, 1]
#         if tf.__version__ <= '0.11':
#             bgr = tf.concat(3, [
#                 blue - VGG_MEAN[0],
#                 green - VGG_MEAN[1],
#                 red - VGG_MEAN[2],
#             ])
#         else:
#             bgr = tf.concat(
#                 [
#                     blue - VGG_MEAN[0],
#                     green - VGG_MEAN[1],
#                     red - VGG_MEAN[2],
#                 ], axis=3)
#         assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
#         """ input layer """
#         net_in = InputLayer(bgr, name='input')
#         """ conv1 """
#         network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
#         network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
#         """ conv2 """
#         network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
#         network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
#         """ conv3 """
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
#         """ conv4 """
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
#         conv = network
#         """ conv5 """
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
#         """ fc 6~8 """
#         network = FlattenLayer(network, name='flatten')
#         network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
#         network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
#         network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
#         print("build model finished: %fs" % (time.time() - start_time))
#         return network, conv


# def vgg16_cnn_emb(t_image, reuse=False):
#     """ t_image = 244x244 [0~255] """
#     with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
#         tl.layers.set_name_reuse(reuse)
#
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#         net_in = InputLayer(t_image - mean, name='vgg_input_im')
#         """ conv1 """
#         network = tl.layers.Conv2dLayer(net_in,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool1')
#         """ conv2 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool2')
#         """ conv3 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool3')
#         """ conv4 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_3')
#
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool4')
#         conv4 = network
#
#         """ conv5 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool5')
#
#         network = FlattenLayer(network, name='vgg_flatten')
#
#         # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
#         # new_network = tl.layers.DenseLayer(network, n_units=4096,
#         #                     act = tf.nn.relu,
#         #                     name = 'vgg_out/dense')
#         #
#         # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
#         # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
#         #             b_init=None, name='vgg_out/out')
#         return conv4, network
