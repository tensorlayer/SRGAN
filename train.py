#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D
from config import config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128

ni = int(np.sqrt(batch_size))

def train():
    # create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img
    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        return lr_patch, hr_patch
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    # value = train_ds.make_one_shot_iterator().get_next()

    # obtain models
    G = get_G((batch_size, None, None, 3)) # (None, 96, 96, 3)
    D = get_D((batch_size, None, None, 3)) # (None, 384, 384, 3)
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    print(G)
    print(D)
    print(VGG)

    # G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode'])) # in case you want to restore a training
    # D.load_weights(checkpoint_dir + '/d_{}.h5'.format(tl.global_flag['mode']))

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(mse_loss, var_list=g_vars)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(d_loss, var_list=d_vars)

    G.train()
    D.train()
    VGG.train()

    # initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)
    for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
        step_time = time.time()
        with tf.GradientTape() as tape:
            fake_hr_patchs = G(lr_patchs)
            mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
        grad = tape.gradient(mse_loss, G.trainable_weights)
        g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
        step += 1
        epoch = step//n_step_epoch
        print("Epoch: [{}/{}] step: [{}/{}] time: {}s, mse: {} ".format(
            epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_init_{}.png'.format(epoch))

    # adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
        with tf.GradientTape(persistent=True) as tape:
            fake_patchs = G(lr_patchs)
            logits_fake = D(fake_patchs)
            logits_real = D(hr_patchs)
            feature_fake = VGG((fake_patchs+1)/2.)
            feature_real = VGG((hr_patchs+1)/2.)
            d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
            d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
            d_loss = d_loss1 + d_loss2
            g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
            mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
            vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
            g_loss = mse_loss + vgg_loss + g_gan_loss
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
        grad = tape.gradient(d_loss, D.trainable_weights)
        d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
        step += 1
        epoch = step//n_step_epoch
        print("Epoch: [{}/{}] step: [{}/{}] time: {}s, g_loss(mse:{}, vgg:{}, adv:{}) d_loss: {}".format(
            epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_{}.png'.format(epoch))
            G.save_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))
            D.save_weights(checkpoint_dir + '/d_{}.h5'.format(tl.global_flag['mode']))

    # ### old
    # ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    # t_target_image_224 = tf.image.resize(
    #     t_target_image, size=[224, 224], method=0,
    #     align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    # t_predict_image_224 = tf.image.resize(net_g.outputs, size=[224, 224], method=0)
    #
    # net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    # _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)
    #
    # # ###========================== DEFINE TRAIN OPS ==========================###
    # d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    # d_loss = d_loss1 + d_loss2
    #
    # g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    # mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    # vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    #
    # g_loss = mse_loss + vgg_loss + g_gan_loss
    #
    # g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    # d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    #
    # with tf.variable_scope('learning_rate'):
    #     lr_v = tf.Variable(lr_init, trainable=False)
    # ## Pretrain
    # g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    # ## SRGAN
    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    # d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    #
    # ###========================== RESTORE MODEL =============================###
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # tl.layers.initialize_global_variables(sess)
    # if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
    #     tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)
    #
    #
    # ###============================= TRAINING ===============================###
    # ## use first `batch_size` of train set to have a quick test during training
    # sample_imgs = train_hr_imgs[0:batch_size]
    # # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    # sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    # print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    # sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    # print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    # tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    # tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    # tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_96.png')
    # tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')
    #
    # ###========================= initialize G ====================###
    # ## fixed learning rate
    # sess.run(tf.assign(lr_v, lr_init))
    # print(" ** fixed learning rate: %f (for init G)" % lr_init)
    # for epoch in range(0, n_epoch_init + 1):
    #     epoch_time = time.time()
    #     total_mse_loss, n_iter = 0, 0
    #
    #     ## If your machine cannot load all images into memory, you should use
    #     ## this one to load batch of images while training.
    #     # random.shuffle(train_hr_img_list)
    #     # for idx in range(0, len(train_hr_img_list), batch_size):
    #     #     step_time = time.time()
    #     #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
    #     #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
    #     #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
    #     #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
    #
    #     ## If your machine have enough memory, please pre-load the whole train set.
    #     for idx in range(0, len(train_hr_imgs), batch_size):
    #         step_time = time.time()
    #         b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
    #         b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
    #         ## update G
    #         errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
    #         print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
    #         total_mse_loss += errM
    #         n_iter += 1
    #     log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
    #     print(log)
    #
    #     ## quick evaluation on train set
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
    #         print("[*] save images")
    #         tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
    #
    #     ## save model
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
    #
    # ###========================= train GAN (SRGAN) =========================###
    # for epoch in range(0, n_epoch + 1):
    #     ## update learning rate
    #     if epoch != 0 and (epoch % decay_every == 0):
    #         new_lr_decay = lr_decay**(epoch // decay_every)
    #         sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
    #         log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
    #         print(log)
    #     elif epoch == 0:
    #         sess.run(tf.assign(lr_v, lr_init))
    #         log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
    #         print(log)
    #
    #     epoch_time = time.time()
    #     total_d_loss, total_g_loss, n_iter = 0, 0, 0
    #
    #     ## If your machine cannot load all images into memory, you should use
    #     ## this one to load batch of images while training.
    #     # random.shuffle(train_hr_img_list)
    #     # for idx in range(0, len(train_hr_img_list), batch_size):
    #     #     step_time = time.time()
    #     #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
    #     #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
    #     #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
    #     #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
    #
    #     ## If your machine have enough memory, please pre-load the whole train set.
    #     for idx in range(0, len(train_hr_imgs), batch_size):
    #         step_time = time.time()
    #         b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
    #         b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
    #         ## update D
    #         errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
    #         ## update G
    #         errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
    #         print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
    #               (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
    #         total_d_loss += errD
    #         total_g_loss += errG
    #         n_iter += 1
    #
    #     log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
    #                                                                             total_g_loss / n_iter)
    #     print(log)
    #
    #     ## quick evaluation on train set
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
    #         print("[*] save images")
    #         tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
    #
    #     ## save model
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
    #         tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)


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
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    G = get_G([1, None, None, 3])
    # G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))
    G.load_weights("g_srgan.npz")
    G.eval()
    
    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
