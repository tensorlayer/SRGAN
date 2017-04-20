from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 0.0001
config.TRAIN.beta1 = 0.9
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 10
config.TRAIN.n_epoch = 100      # 10^5 update iterations

config.TRAIN.img_path = '/media/gyang/RAIDARRAY/Data/SuperResolutionDatasets/DIV2K_train_HR/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
