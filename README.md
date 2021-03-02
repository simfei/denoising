## Multiphoton Microscopy Image Denoising with Deep Learning
---

Multiphoton microscopy (MPM) images are captured inherently with low signal-to-noise ratio (SNR), inhibiting
processes to image deeper brain layer, achieve higher time and spatial resolution. While classical
approaches based on linear filtering fail to deal with the dominated Poisson noise in MPM images, image
restoration with deep learning is currently a hot topic. In this work, three supervised (CARE, DnCNN, and ResNet) and three unsupervised (Noise2Noise, Noise2Void, and Probabilistic Noise2Void) deep learning methods are compared in the denoising performance of MPM images.The gap between supervised and unsupervised methods is investigated. By adding images with different noise levels in training data, our models are expected with generalization to blind-noise images. The ability of generalization
is also examined by bias-free neural networks. Results have shown that our deep learning based models achieved satisfying denoising performance, with generalization across a broad range of noise levels. It is also proved that unsupervised methods only show slightly degraded denoising performance compared to supervised methods. This finding is of significant meaning in that collecting experimental images for training data is expensive and sometimes difficult. Unsupervised methods only need pairs of noisy images or even single noisy images to train the networks. These methods are more suitable for denoising a time series of the same sample with sensible motion, due to the fact that ground truth images used in training supervised models are usually not well aligned with input noisy images. Thus supervised algorithms tend to get the same restored image for all frames in a time series, which would cause loss of interested details, such as blood flow. Further applications by our denoising models also showed promising results, such as denoising third harmonics generation (THG) images captured by commonly used lasers and better ratiometric imaging results after denoising by our models.

## Training
---

### 1. Installation for training on GPU

The implementation of CARE, N2N and N2V requires tensorflow version1. The implementation of DnCNN and ResNet requires torch. 

**For training on GPU**, first install [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then install the required version of CUDA and CuDNN. In our case, we use CUDA 9.0 and CuDNN 7.1.4.

Two enviroments are separately created for tensorflow and torch. The following lines show how to create an enviroment and install the packages:
``` 
$ conda create -n tensorflow python==3.6
$ conda activate tensorflow
$ conda install tensorflow-gpu==1.12.0
$ pip install n2v
$ pip install csbdeep
$ pip install numpy==1.19.0
$ pip install tifffile
```

``` 
$ conda create -n torch python==3.6
$ conda activate denoising
$ pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install numpy=1.19.1
$ pip install tifffile
```

### 2. How to use the train script

The arguments in parser are as follows:
```
--model: choose the model for training. 'care', 'dncnn', 'resnet', 'n2n', 'n2v', default='n2v'.
--dataDir: where the training data is, default='data/'
--numImgsInTif: how many frames in a .tif file are chosen for training, default=1.
--expandData: whether to expand data by averaging 2,4,8,16 raw images, default=True.
--shape: the shape of input patches, default=(256,256).
--testSize: the size of validation set, default=0.1.
--augment: whther to do data augmentatioin, default=True.
--batchSize: training batch size, default=16.
--epochs: training epochs, default=100.
--lr: learning rate, default=0.0004.
--modelName: name of the trained model to be saved, default='n2v'.
--baseDir: the directory to save the model, default='save_models/'
--bias: whether to use bias in network, only used for DnCNN and U-Net, default=False.
--structN2VMask: blind mask for structure Noise2Void, default=None.
```

Go to the directory with script.py, and activate the required environment.
```
conda activate torch   
```
or
```
conda activate tensorflow
```
Run the following command for training. The trained model will be saved to baseDir. The command is an example, you can change or add the arguments.
```
python train.py --model=n2v --dataDir=data/ batchSize=128 --epochs=200 --modelName=n2v --baseDir=save_models/
```

## Prediction
---

### 1. Installation for prediction on CPU

Firstly install [Anaconda](https://docs.anaconda.com/anaconda/install/windows/). The following lines show how to create an environment and add the environment to jupyter:

```
conda create -n [ENV_NAME] python==3.6
conda activate [ENV_NAME]
conda install -n [ENV_NAME] ipykernel
python -m ipykernel install --user --name [ENV_NAME] --display-name [DISPLAY_NAME]
```

The requirements.txt file is for prediction. Go to the directory with requirements.txt, run the following command to install all dependencies
```
pip install -r requirements.txt
```

### 2. How to use the predict script

The arguments in parser are as follows:

```
--imgType: the type of image. 'vas', 'ast', 'neu' and 'smp', default='ast'.
--modelName: which model you want to use. 'care', 'dncnn', 'resnet', 'n2n' and 'n2v', default='n2v'.
--imgPath: directory in which test images are.
--savePath: where to save denoised images, default='denoised'
```

Go to the directory with predict.py, run the following command. Denoised images will be saved to savePath. This command is an example, you can change the arguments.
```
(conda activate [ENV_NAME])
python script.py --imgType=vas --modelName=n2v --imgPath=test_images/VAS/raw/ --savePath=denoised/
```

### 3. How to use jupyter notebook for prediction and visualization

Open predict.ipynb in notebook, select the kernel you create, change arguments in the Config part and run the cells.
