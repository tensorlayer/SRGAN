#! /usr/bin/python
# -*- coding: utf8 -*-


import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## adversarial learning
n_epoch = config.TRAIN.n_epoch
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
lr_decay_init = config.TRAIN.lr_decay_init
decay_every_init = config.TRAIN.decay_every_init

ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=32):
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    ## create folders to save result images and trained model
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)

    SRGAN_d = SRGAN_d2
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)
    # exit()

    ## vgg inference
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)
    # print(vgg_predict_emb.outputs)
    # # exit()

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    pixel_mse = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_mse = tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True) # perceptual loss
    g_loss = (pixel_mse + vgg_mse) + 1e-3 * g_gan_loss
    # g_loss = pixel_mse + 1e-3 * g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## We employed the trained MSE-based SRResNet network as initialization for
    ## the generator when training the actual GAN to avoid undesired local optima.
    ## All SRGAN variants were trained with 10^5 update iterations at a learning
    ## rate of 10−4 and another 10-5 iterations at a lower rate of 10−5. We
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(pixel_mse, var_list=g_vars) # pre-train G

    ## We alternate updates to the generator and discriminator network, which is
    ## equivalent to k = 1 as used in Goodfellow et al. [21]. Our generator
    ## network has 16 identical (B = 16) residual blocks.
    ## For each mini-batch we crop 16 random 96 × 96 sub images of distinct
    ## training images. Note that we can apply the generator model to images of
    ## arbitrary size as it is fully convolutional. For optimization we use
    ## Adam with β1 = 0.9. The SRResNet networks were trained with a learning
    ## rate of 10−4 and 10^6 update iterations.
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}_init.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print(sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print(sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir+'/_train_sample_384.png')

    for epoch in range(0, n_epoch + n_epoch_init):
        ## update learning rate
        if epoch < n_epoch_init:
            if epoch !=0 and (epoch % decay_every_init == 0):
                new_lr_decay = lr_decay_init ** (epoch // decay_every_init)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate (for init G): %f" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                log = " ** init G, lr: %f  decay_every_init: %d, lr_decay_init: %f (for init G)" % (lr_init, decay_every_init, lr_decay_init)
                print(log)
        elif epoch == n_epoch_init:
            sess.run(tf.assign(lr_v, lr_init))
            print(" ** fine-tune with Content loss + Adversarial loss, fixed lr: %f" % lr_init)
        # else:
        #     if epoch != n_epoch_init and (epoch % decay_every == 0):
        #         new_lr_decay = lr_decay ** (epoch // decay_every)
        #         sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        #         log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        #         print(log)
        #     elif epoch == n_epoch_init:
        #         sess.run(tf.assign(lr_v, lr_init))
        #         log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
        #         print(log)

        epoch_time = time.time()

        total_d_loss, total_g_loss, total_mse_loss, n_iter = 0, 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            if epoch < n_epoch_init:
                ## initalization G with pixel-MSE only
                errM, _ = sess.run([pixel_mse, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
            else:
                ## update D
                errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                ## update G
                errG, errM, errV, errA, _ = sess.run([g_loss, pixel_mse, vgg_mse, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_d_loss += errD
                total_g_loss += errG
            n_iter += 1

        if epoch < n_epoch_init:
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_mse_loss/n_iter)
            print(log)
        else:
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
            print(log)

        ###======================= EVALUATION =========================###
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 5 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
            print('out', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir+'/train_%d.png' % epoch)

        ###======================= SAVE MODEL =========================###
        if (epoch != 0) and (epoch % 5 == 0):
            if epoch < n_epoch_init:
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
            else:
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    valid_lr_img = valid_lr_imgs[0]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale
    # print(valid_lr_img.min(), valid_lr_img.max())
    # exit()

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan_init.npz', network=net_g)

    ###======================= EVALUATION =============================###
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})

    print("LR size: %s /  gen HR size: %s" % (size, out.shape))
    print('out', out.shape, out.min(), out.max())
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir+'/valid_gen.png')
    tl.vis.save_image(valid_lr_imgs[0], save_dir+'/valid_lr.png')
    tl.vis.save_image(valid_hr_imgs[0], save_dir+'/valid_hr.png')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
