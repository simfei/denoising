import os
from os import path
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="basedir of models for each type of image", default="models/models_ast/")
parser.add_argument("--modelName", help="which model you want to use: "
                                        "care, dncnn, resnet, n2n, n2v, pn2v and struct_n2v_xxx", default='n2v')
parser.add_argument("--imgPath", help="path of test images", default="test_images/AST/raw/")
parser.add_argument("--savePath", help="path to save denoised images", default="denoised/")
parser.add_argument("--multiFrames", help="whether there are multi frames in an image", default='False')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# check args
args = parser.parse_args()
if args.multiFrames == 'False':
    multi_frames = False
else:
    multi_frames = True
basedir = args.baseDir
model_name = args.modelName
test_img_path = args.imgPath
save_path = args.savePath

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
            preds[i,...] = prediction.tiledPredict(imgs[i], model, ps=256, overlap=48, device=device, noiseModel=None)
    else:
        raise Exception("Wrong model name!")
    return preds

# load test images
test_imgs = []
files = os.listdir(test_img_path)
for file in files:
    print('loading ' + file)
    image = imread(test_img_path + file)
    max_value = np.percentile(image, 99.5)
    test_imgs.append(cut_off(image, max_value))
assert len(test_imgs) == len(files)
if multi_frames is False:
    test_imgs = np.stack(test_imgs)
print('done')

# load nets
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
since = time.time()
if multi_frames is False: 
    preds = custom_predict(test_imgs, model_name)
    preds = np.clip(preds, 0, 1)
    assert preds.shape[0] == len(files)
    for i, file in enumerate(files):
        imsave(save_path + file, preds[i])
else:
    for i in range(len(test_imgs)):
        pred = custom_predict(test_imgs[i], model_name)
        pred = np.clip(pred, 0, 1)
        if pred.shape[0] == 1:
            pred = pred[0]
        imsave(save_path + files[i], pred)

print('Finished denoising. Time: %.2fs'%(time.time()-since))
print("Denoised images are saved to {}".format(save_path))
