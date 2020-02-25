## Super Resolution Examples




We run this script under [TensorFlow](https://www.tensorflow.org) 2.0 and the [TensorLayer](https://github.com/tensorlayer/tensorlayer) 2.0+. For TensorLayer 1.4 version, please check [release](https://github.com/tensorlayer/srgan/releases).

<!---
⚠️ This repo will be merged into example folder of [tensorlayer](https://github.com/zsdonghao/tensorlayer) soon.
-->
🚀🚀🚀🚀🚀🚀 **THIS PROJECT WILL BE CLOSED AND MOVED TO [THIS FOLDER](https://github.com/tensorlayer/tensorlayer/tree/master/examples) IN A MONTH.**

🚀🚀🚀🚀🚀🚀 **THIS PROJECT WILL BE CLOSED AND MOVED TO [THIS FOLDER](https://github.com/tensorlayer/tensorlayer/tree/master/examples) IN A MONTH.**

🚀🚀🚀🚀🚀🚀 **THIS PROJECT WILL BE CLOSED AND MOVED TO [THIS FOLDER](https://github.com/tensorlayer/tensorlayer/tree/master/examples) IN A MONTH.**

<!--More cool Computer Vision applications such as pose estimation and style transfer can be found in this [organization](https://github.com/tensorlayer).**
-->

### SRGAN Architecture

TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/model.jpeg" width="80%" height="10%"/>
</div>
</a>


### Results

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/SRGAN_Result2.png" width="80%" height="50%"/>
</div>
</a>

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/SRGAN_Result3.png" width="80%" height="50%"/>
</div>
</a>

### Prepare Data and Pre-trained VGG

- 1. You need to download the pretrained VGG19 model in [here](https://github.com/tensorlayer/pretrained-models/tree/master/models) as [tutorial_models_vgg19.py](https://github.com/tensorlayer/tensorlayer/blob/master/examples/pretrained_cnn/tutorial_models_vgg19.py) show.
- 2. You need to have the high resolution images for training.
  -  In this experiment, I used images from [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/), so the hyper-paremeters in `config.py` (like number of epochs) are seleted basic on that dataset, if you change a larger dataset you can reduce the number of epochs. 
  -  If you dont want to use DIV2K dataset, you can also use [Yahoo MirFlickr25k](http://press.liacs.nl/mirflickr/mirdownload.html), just simply download it using `train_hr_imgs = tl.files.load_flickr25k_dataset(tag=None)` in `main.py`. 
  -  If you want to use your own images, you can set the path to your image folder via `config.TRAIN.hr_img_path` in `config.py`.



### Run
- Set your image folder in `config.py`, if you download [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) dataset, you don't need to change it. 
- Other links for DIV2K, in case you can't find it : [test\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip), [train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [train\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip), [valid_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR.zip), [valid\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip).

```python
config.TRAIN.img_path = "your_image_folder/"
```

- Start training.

```bash
python train.py
```

- Start evaluation. 

<!--([pretrained model](https://github.com/tensorlayer/srgan/releases/tag/1.2.0) for DIV2K)-->

```bash
python train.py --mode=evaluate 
```


### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)

### Author
- [zsdonghao](https://github.com/zsdonghao)

### Citation
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}
```

### Other Projects

- [Style Transfer](https://github.com/tensorlayer/adaptive-style-transfer)
- [Pose Estimation](https://github.com/tensorlayer/openpose)

### Discussion

- [TensorLayer Slack](https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc)
- [TensorLayer WeChat](https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/wechat_group.md)

### License

- For academic and non-commercial use only.
- For commercial use, please contact tensorlayer@gmail.com.
