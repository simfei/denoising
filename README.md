## Multiphoton Microscopy Image Denoising with Deep Learning

Multiphoton microscopy (MPM) images are captured inherently with low signal-to-noise ratio (SNR), inhibiting
processes to image deeper brain layer, achieve higher time and spatial resolution. While classical
approaches based on linear filtering fail to deal with the dominated Poisson noise in MPM images, image
restoration with deep learning is currently a hot topic. In this work, three supervised (CARE, DnCNN, and ResNet) and three unsupervised (Noise2Noise, Noise2Void, and Probabilistic Noise2Void) deep learning methods are compared in the denoising performance of MPM images.The gap between supervised and unsupervised methods is investigated. By adding images with different noise levels in training data, our models are expected with generalization to blind-noise images. The ability of generalization
is also examined by bias-free neural networks. Results have shown that our deep learning based models achieved satisfying denoising performance, with generalization across a broad range of noise levels. It is also proved that unsupervised methods only show slightly degraded denoising performance compared to supervised methods. This finding is of significant meaning in that collecting experimental images for training data is expensive and sometimes difficult. Unsupervised methods only need pairs of noisy images or even single noisy images to train the networks. These methods are more suitable for denoising a time series of the same sample with sensible motion, due to the fact that ground truth images used in training supervised models are usually not well aligned with input noisy images. Thus supervised algorithms tend to get the same restored image for all frames in a time series, which would cause loss of interested details, such as blood flow. Further applications by our denoising models also showed promising results, such as denoising third harmonics generation (THG) images captured by commonly used lasers and better ratiometric imaging results after denoising by our models.

## Training

### 1. Installation for training on GPU

The implementation of CARE, N2N and N2V requires tensorflow. The implementation of DnCNN, ResNet and PN2V requires torch. 

**For training on GPU**, firstly install [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then install the required version of CUDA and CuDNN. In our case, we use CUDA 11.4 and CuDNN 8.2.4.

Two enviroments are separately created for tensorflow and torch. The following lines show how to create an enviroment and install the packages:
``` 
$ conda create -n tensorflow python==3.7.9
$ conda activate tensorflow
$ pip install tensorflow-gpu==2.4.1 
$ pip install keras=2.3.1
$ pip install n2v
$ pip install csbdeep
$ pip install numpy
$ pip install tifffile
```

``` 
$ conda create -n torch python==3.7.9
$ conda activate torch
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install numpy
$ pip install tifffile
```

### 2. How to use the train script

The arguments in parser are as follows:
```
--model: choose the model for training: care, dncnn, resnet, n2n, n2v, or pn2v, default=n2v.
--dataDir: path of the training data, default=data/.
--numImgsInTif: how many frames in a .tif file are chosen for training. Int, default=1.
--expandData: whether to expand data by averaging 2, 4, 8, 16 raw images, default=True.
--shape: the shape of input patches, e.g. input 256, the shape is (256,256). Int, default=256.
--testSize: the size of validation set. Float, default=0.1.
--augment: whther to do data augmentatioin, default=True.
--batchSize: training batch size. Int, default=16.
--epochs: training epochs. Int, default=100.
--lr: learning rate. Float, default=0.0004.
--modelName: name of the trained model to be saved, default=n2v.
             For PN2V, modelName is also the path to save the model.
             For structured N2V, add 'struct' as prefix, e.g. struct_n2v_1x5.
--baseDir: the base directory to save the model, default=save_models/.
--bias: whether to use bias in network, only used for DnCNN and U-Net, default=False.
--structN2VMask: blind mask for structure Noise2Void, choose from 1x3, 1x5, 1x9, 3x1, 5x1, 9x1, default=None.
--unetDepth: depth of UNet, only used for model: care, n2n, n2v, and pn2v. Int, default=2.
```
Noted that for PN2V in this code, the noise model is created by directly using all training data. To train a structured Noise2Void model, choose model 'n2v', add the argument structN2VMask, and save the model with a prefix 'struct' in the argument modelName. For models with UNet architecture, 2-depth UNet is usually enough for training CARE, N2N and N2V. PN2V usually use 3-depth UNet.

Go to the directory with train.py, and activate the required environment.
```
conda activate torch   
```
or
```
conda activate tensorflow
```
Run the following command for training. The trained model will be saved to baseDir. The command is an example, you can change or add the arguments.
```
python train.py --model=n2v --dataDir=data/ --batchSize=128 --epochs=200 --modelName=n2v --baseDir=save_models/
```

## Prediction

### 1. Installation for prediction on CPU

Firstly install [Anaconda](https://docs.anaconda.com/anaconda/install/windows/). Then the following lines show how to create an environment and add the environment to jupyter:

```
conda create -n [ENV_NAME] python==3.7.9
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
--baseDir: basedir of models for each type of image, default=models/models_ast/.
--modelName: which model you want to use: care, dncnn, resnet, n2n, n2v and pn2v, default=n2v.
             (also includes struct_n2v_1x5, struct_n2v_5x1, etc.)
--imgPath: path of test images.
--savePath: path to save denoised images, default=denoised/.
--multiFrames: whether there are multi frames in an image. True or False, default=False.
```

Go to the directory with predict.py, run the following command. Denoised images will be saved to savePath. This command is an example, you can change the arguments.
```
(conda activate [ENV_NAME])
python predict.py --baseDir=models/models_vas --modelName=n2v --imgPath=test_images/VAS/raw/ --savePath=denoised/
```

### 3. How to use the ratiometric script

The arguments in parser are as follows:

```
--baseDir: basedir of models for each type of image, default=models/models_ast/.
--modelName: which model you want to use: care, dncnn, resnet, n2n, n2v and pn2v, default=care.
             (also includes struct_n2v_1x5, struct_n2v_5x1, etc.)
--imgPath: path of test images.
--savePath: path to save denoised images, default=ratiometric/.
--binning: number of frames to average, default=1.
```

Go to the directory with ratiometric.py, run the following command. Denoised images will be saved to savePath. Each denoised image has two channels. This command is an example, you can add or change the arguments.
```
(conda activate [ENV_NAME])
python predict.py --baseDir=models/models_vas --modelName=n2v --imgPath=test_images/VAS/raw/ --binning=1
```

### 3. How to use jupyter notebook for prediction and visualization

Open **predict.ipynb** or **ratiometric.ipynb** in notebook, select the kernel you created, change arguments in the Config part and run the cells.

### Note

Files in __nets/pn2v__ and __nets/unet__ are copies from https://github.com/juglab/pn2v.

### *References*

[1] Weigert, M., Schmidt, U., Boothe, and T. et al. Content-aware image restoration: pushing the limits of fluorescence microscopy. Nat Methods, 15:1090–1097, 2018.

[2] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE Transactions on Image Processing, 26(7):3142–3155, 2017.

[3] Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, and Timo Aila. Noise2noise: Learning image restoration without clean data, 2018.

[4] Alexander Krull, Tim-Oliver Buchholz, and Florian Jug. Noise2void - learning denoising from single noisy images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[5] Alexander Krull, Tomas Vicar, Mangal Prakash, Manan Lalit, and Florian Jug. Probabilistic noise2void: Unsupervised content-aware denoising. Frontiers in Computer Science, 2:5, 2020.

[6] C. Broaddus, A. Krull, M. Weigert, U. Schmidt, and G. Myers. Removing structured noise with selfsupervised blind-spot networks. In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), pages 159–163, 2020.

[7] Sreyas Mohan, Zahra Kadkhodaie, Eero P. Simoncelli, and Carlos Fernandez-Granda. Robust and interpretable blind image denoising via bias-free convolutional neural networks, 2020.
