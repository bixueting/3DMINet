# Sphere-Guided 3D Microwave Imaging Network as Point Cloud Shape with Offset-attention Moudule (TGRS)

This repository contains a Pytorch implementation of the paper:

[Sphere-Guided 3D Microwave Imaging Network as Point Cloud Shape with Offset-attention Moudule](https://liruihui.github.io/publication/SP-GAN/). 
<br>
[Xueting Bi](https://bixueting.github.io/).
<br>
<br>

![teaser](figures/correspondence.png)

### Dependencies
* Python 3.6
* CUDA 10.0.
* [PyTorch](http://pytorch.org/). Codes are tested with version 1.2.0
* (Optional) [TensorboardX](https://www.tensorflow.org/) for visualization of the training process. 

Following is the suggested way to install these dependencies: 
```bash
# Create a new conda environment
conda create -n wp python=3.6
conda activate wp

# Install pytorch (please refer to the commend in the official website)
conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch -y
```

### Usage

To train a model on point clouds sampled from 3D shapes:

    python train.py

Log files and network parameters will be saved to `log` folder in default. 


We provide various visulization function for shape interpolation, part interpolation, and so on. 

    python visual.py

### Evaluation


### Citation




