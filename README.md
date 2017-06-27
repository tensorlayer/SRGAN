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
- In this experiment, we used images from [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/), besides [Yahoo MirFlickr25k](http://press.liacs.nl/mirflickr/mirdownload.html) is also a good choice. Alternatively, you can use your own data by setting your image folder in `config.py`.

- Download VGG model as [tutorial_vgg19.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py) show.



### Run
- Set your image folder in `config.py`.

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

