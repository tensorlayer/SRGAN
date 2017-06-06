from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 0.0001
config.TRAIN.beta1 = 0.9
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 100
config.TRAIN.n_epoch = 400      # 10^5 update iterations

config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()

config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
