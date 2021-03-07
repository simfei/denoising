import os
from os import path
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="basedir of models for each type of image", default="models/models_ast/")
parser.add_argument("--modelName", help="which model you want to use", default='care')
parser.add_argument("--imgPath", help="the path of test images", default="test_images/AST/ratio/")
parser.add_argument("--savePath", help="the path to save denoised images", default="ratiometric/")
parser.add_argument("--binning", help="number of frames to average", default=1)

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# check args
args = parser.parse_args()
basedir = args.baseDir
model_name = args.modelName
test_img_path = args.imgPath
save_path = args.savePath
binning = int(args.binning)
if not basedir.endswith('/'):
    basedir = basedir+'/'
if not test_img_path.endswith('/'):
    test_img_path = test_img_path+'/'
if not save_path.endswith('/'):
    save_path = save_path+'/'
if not path.exists(save_path):
    os.mkdir(save_path)

# import all dependencies
import numpy as np
from tifffile import imread, imsave
import time

import torch
from csbdeep.models import CARE
from n2v.models import N2V
from nets.dncnn import DnCNN
from nets.resnet import ResNet

import sys
sys.path.append("nets")
import unet
from pn2v import prediction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define functions
def cut_off(img_, max_value):
    img = np.copy(img_)
    img = img.astype('float32')
    mask = img > max_value
    img[mask] = max_value
    img = img / max_value * 1.0
    return img

def custom_predict(imgs, model_name):
    preds = None
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis,...]
    if model_name in ['care', 'n2n']:
        imgs = imgs[...,np.newaxis]
        preds = model.keras_model.predict(imgs)[...,0]
    elif model_name == 'n2v' or model_name.startswith('struct'):
        preds = np.zeros(imgs.shape)
        for i in range(imgs.shape[0]):
            preds[i,...] = model.predict(imgs[i], axes='YX')
    elif model_name in ['dncnn', 'resnet']:
        imgs = np.expand_dims(imgs, axis=1)
        imgs = torch.as_tensor(imgs.astype('float32'))
        preds = model(imgs).detach().numpy()[:,0,...]
    elif model_name == 'pn2v':
        imgs = imgs.astype('float32')
        preds = np.zeros(imgs.shape)
        for i in range(imgs.shape[0]):
            preds[i,...] = prediction.tiledPredict(imgs[i], model, ps=256, overlap=48, device=None, noiseModel=None)
    else:
        raise Exception("Wrong model name!")
    return preds

# load images
files = os.listdir(test_img_path)
ch1 = []
ch2 = []
max_values1 = []
max_values2 = []

for file in files:
    print('loading ' + file)
    image = imread(test_img_path + file)
    b = 0
    raws1 = []
    raws2 = []
    for i in range(image.shape[0]):
        if b < 2*binning:
            if i%2 == 0:
                raws1.append(image[i])
            else:
                raws2.append(image[i])
            b += 1
    assert len(raws1) == len(raws2)
    if len(raws1) == 1:
        raws1 = raws1[0][np.newaxis,...]
        raws2 = raws2[0][np.newaxis,...]
    else:
        raws1 = np.stack(raws1)
        raws2 = np.stack(raws2)
    max_value1 = np.percentile(raws1, 99.5)
    max_value2 = np.percentile(raws2, 99.5)
    raws1 = cut_off(raws1, max_value1)
    raws2 = cut_off(raws2, max_value2)
    ch1.append(np.mean(raws1, axis=0))
    ch2.append(np.mean(raws2, axis=0))
    max_values1.append(max_value1)
    max_values2.append(max_value2)
assert len(ch1) == len(ch2)
assert len(max_values1) == len(max_values2)
assert len(ch1) == len(max_values1)
assert len(ch1) == len(files)
print('done')

# load model
model = None
if model_name == 'care':
    print('loading CARE.')
    model = CARE(config=None, name='care', basedir=basedir)
if model_name == 'n2n':
    print('loading N2N.')
    model = CARE(config=None, name='n2n', basedir=basedir)
if model_name == 'n2v':
    print('loading N2V.')
    model = N2V(config=None, name='n2v', basedir=basedir)
if model_name.startswith('struct'):
    print('loading {}'.format(model_name))
    model = N2V(config=None, name=model_name, basedir=basedir)
if model_name == 'pn2v':
    print('loading PN2V.')
    model = torch.load('{}/pn2v/save_model/last_PN2V.net'.format(basedir), map_location=torch.device('cpu'))
if model_name == 'dncnn':
    print('loading DnCNN.')
    model = DnCNN(bias=False)
    model.load_state_dict(torch.load('{}/dncnn/dncnn.pt'.format(basedir), map_location=device))
    model.eval()
if model_name == 'resnet':
    print('loading ResNet.')
    model = ResNet()
    model.load_state_dict(torch.load('{}/resnet/resnet.pt'.format(basedir), map_location=device))
    model.eval()

# denoise and save denoised images
# each denoised image has two channels
since = time.time()

for i in range(len(ch1)):
    pred1 = custom_predict(ch1[i], model_name)[0]
    pred2 = custom_predict(ch2[i], model_name)[0]
    pred1 = np.clip(pred1, 0, 1)*max_values1[i]
    pred2 = np.clip(pred2, 0, 1)*max_values2[i]
    pred = np.stack([pred1, pred2])
    imsave(save_path+files[i], pred)
print('Finished denoising. Time: %.2fs'%(time.time()-since))
print("Denoised images are saved to {}".format(save_path))
