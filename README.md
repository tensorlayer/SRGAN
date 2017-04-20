## Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

We run this script under TensorFlow 1.0.1 and the self-contain TensorLayer. If you got error, you may need to update TensorLayer.

### Model
<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/model.jpeg" width="70%" height="30%"/>
</div>
</a>


### Results
<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/result.png" width="70%" height="30%"/>
</div>
</a>

### Run
1. Set your image folder in `config.py`.

```
config.TRAIN.img_path = "your_image_folder/"
```
2. Start training.

```
python train.py
```


### Citation
If you find this script useful or want to use it for academic, please cite:

```
@article{xxx,
  title={XXX},
  author={H. Dong},
  journal={arXiv preprint arXiv},
  year={2017}
}
```

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)