## Super Resolution Examples

We run this script under [TensorFlow](https://www.tensorflow.org) 1.2 and the self-contained [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/). If you got error, you may need to update TensorLayer.


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

### Prepare Data and Pre-trained VGG

- 1. You need to download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) as [tutorial_vgg19.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py) show.
- 2. You need to have the high resolution images for training.
  -  In this experiment, I used images from [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/), so the hyper-paremeters in `config.py` (like number of epochs) are seleted basic on that dataset, if you change a larger dataset you can reduce the number of epochs. 
  -  If you dont want to use DIV2K dataset, you can also use [Yahoo MirFlickr25k](http://press.liacs.nl/mirflickr/mirdownload.html), just simply download it using `train_hr_imgs = tl.files.load_flickr25k_dataset(tag=None)` in `main.py`. 
  -  If you want to use your own images, you can set the path to your image folder via `config.TRAIN.hr_img_path` in `config.py`.



### Run
- Set your image folder in `config.py`, if you download [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) dataset, you dont need to change it.

```python
config.TRAIN.img_path = "your_image_folder/"
```

- Start training.

```bash
python main.py
```

- Start evaluation.

```bash
python main.py --mode=evaluate 
```


### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)

