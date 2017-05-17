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


batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
n_epoch = config.TRAIN.n_epoch

img_path = config.TRAIN.img_path

ni = int(np.sqrt(batch_size))


def train():
    # create folders
    save_dir = "samples/{}".format(tl.global_flag['model'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    tl.files.exists_or_mkdir(checkpoint_dir)
    #
    # # logging
    # log_dir = "log_miccai_segc1_pixel_vgg"
    # tl.files.exists_or_mkdir(log_dir)
    #
    # current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    # log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    # log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))
    #
    # log_all = logging.getLogger('log_all')
    # log_all.setLevel(logging.DEBUG)
    # log_all.addHandler(logging.FileHandler(log_all_filename))
    #
    # log_eval = logging.getLogger('log_eval')
    # log_eval.setLevel(logging.INFO)
    # log_eval.addHandler(logging.FileHandler(log_eval_filename))

    with tf.Graph().as_default():
        ##========================== PREPARE DATA ============================##
        img_list = sorted(tl.files.load_file_list(path=img_path, regx='.*.png', printable=False))
        print(img_list)

        ##========================== DEFINE MODEL ============================##
        t_image = tf.placeholder('float32', [batch_size, 64, 64, 3], name='t_image')
        t_target_image = tf.placeholder('float32', [batch_size, 256, 256, 3], name='t_target_image')

        net_g = super_resolution_64_256(t_image, is_train=True, reuse=False)

        net_d, logits_real = super_resolution_256_discriminator(t_target_image, is_train=True, reuse=False)
        _,     logits_fake = super_resolution_256_discriminator(net_g.outputs, is_train=True, reuse=True)

        t_target_image_244 = tf.image.resize_images(t_target_image, size=[244, 244], method=0, align_corners=False) # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_244 = tf.image.resize_images(net_g.outputs, size=[244, 244], method=0, align_corners=False)
        vgg_target_emb, _ = vgg16_cnn_emb(t_target_image_244, reuse=False)
        vgg_predict_emb, _ = vgg16_cnn_emb(t_predict_image_244, reuse=True)

        # evaluation
        net_g_test = super_resolution_64_256(t_image, is_train=False, reuse=True)

        ##========================== DEFINE TRAIN ============================##
        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        d_loss = d_loss1 + d_loss2

        g_gan_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
        pixel_mse = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
        vgg_mse = tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True) # perceptual loss
        # g_loss = (pixel_mse + vgg_mse) + 1e-3 * g_gan_loss
        g_loss = pixel_mse + 1e-3 * g_gan_loss

        g_vars = tl.layers.get_variables_with_name('super_resolution_64_256', True, True)
        d_vars = tl.layers.get_variables_with_name('super_resolution_256_discriminator', True, True)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optim_pixel_mse = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(pixel_mse, var_list=g_vars)

        ##========================== RESTORE MODEL ============================#
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)

        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['model']), network=net_g)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['model']), network=net_d)

        if not os.path.isfile("vgg16_weights.npz"):
            print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
            exit()
        npz = np.load('vgg16_weights.npz')
        assign_op = []
        for idx, val in enumerate(sorted(npz.items())[0:len(vgg_target_emb.all_params)]):
            print("  Loading pretrained VGG16, CNN part %s" % str(val[1].shape))
            assign_op.append(vgg_target_emb.all_params[idx].assign(val[1]))
        sess.run(assign_op)
        vgg_target_emb.print_params(False)

        ##========================== TRAINING ================================##
        sample_imgs = tl.prepro.threading_data(img_list[0:batch_size], fn=get_image_fn, path=img_path)
        sample_imgs_256 = tl.prepro.threading_data(sample_imgs, fn=prepare_train_fn, is_random=False)
        print(sample_imgs_256.shape, sample_imgs_256.min(), sample_imgs_256.max())
        sample_imgs_64 = tl.prepro.threading_data(sample_imgs_256, fn=downsample_fn)
        print(sample_imgs_64.shape, sample_imgs_64.min(), sample_imgs_64.max())
        tl.visualize.save_images(sample_imgs_64, [ni, ni], save_dir+'/_sample_imgs_64.png')
        tl.visualize.save_images(sample_imgs_256, [ni, ni], save_dir+'/_sample_imgs_256.png')

        img_list = img_list[batch_size:]

        n_epoch_init = 20

        for epoch in range(0, n_epoch):
            if epoch !=0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
                print(log)

            epoch_time = time.time()
            random.shuffle(img_list)
            total_d_loss, total_g_loss, total_mse_loss, n_iter = 0, 0, 0, 0
            for idx in range(0, len(img_list), batch_size):
                step_time = time.time()
                b_imgs_list = img_list[idx:idx+batch_size]
                b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_image_fn, path=img_path)
                b_imgs_256 = tl.prepro.threading_data(b_imgs, fn=prepare_train_fn)
                b_imgs_64 = tl.prepro.threading_data(b_imgs_256, fn=downsample_fn)

                # out = sess.run(t_target_image_244, {t_target_image: b_imgs_256})
                # print(b_imgs_256.shape, b_imgs_256.min(), b_imgs_256.max()) (16, 256, 256, 3) -1.0 1.0
                # print(out.shape, out.min(), out.max())        (16, 244, 244, 3) -1.0 1.0
                # exit()

                if epoch < n_epoch_init:
                    # update G via pixel MSE only
                    errM, _ = sess.run([pixel_mse, g_optim_pixel_mse], {t_image: b_imgs_64, t_target_image: b_imgs_256})
                    print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch, n_iter, time.time() - step_time, errM))
                    total_mse_loss += errM
                else:
                    # update D
                    errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_64, t_target_image: b_imgs_256})
                    # update G
                    errG, errM, errV, errA, _ = sess.run([g_loss, pixel_mse, vgg_mse, g_gan_loss, g_optim], {t_image: b_imgs_64, t_target_image: b_imgs_256})
                    # print(errM, errV, errA)
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
            # log_all.debug(log)

            # evaluation
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_64})
            print(out.shape, out.min(), out.max())
            print("[*] save images")
            tl.visualize.save_images(out, [ni, ni], save_dir+'/train_%d.png' % epoch)

            # save model
            if (epoch != 0) and (epoch % 5 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['model']), sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['model']), sess=sess)
                if epoch == (n_epoch_init - 1):
                    tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['model']), sess=sess)
                    tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}_init.npz'.format(tl.global_flag['model']), sess=sess)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='super', help='super')

    args = parser.parse_args()

    tl.global_flag['model'] = args.model

    if tl.global_flag['model'] == 'super':
        train()
    else:
        raise Exception("Unknow --model")
