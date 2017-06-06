## Super Resolution Examples

We run this script under TensorFlow 1.1 and the self-contained TensorLayer. If you got error, you may need to update TensorLayer.


### 1. SRGAN

TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/model.jpeg" width="80%" height="10%"/>
</div>
</a>


#### 1.1 Results
<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/result.png" width="10%" height="10%"/>
</div>
</a>

#### 1.2 Run
1. Set your image folder in `config.py`.

```
config.TRAIN.img_path = "your_image_folder/"
```
2. Start training.

```
python train.py
```

### 2. Fast-SRGAN
 our work


### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)

### Citation
If you find this script useful or want to use it for academic, please cite:

```
@article{xxx,
  title={TensorLayer},
  author={H. Dong},
  journal={arXiv preprint arXiv},
  year={2017}
}
```