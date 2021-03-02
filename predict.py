import os
from os import path
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--imgType", help="the type of image", default="ast")
parser.add_argument("--modelName", help="which model you want to use", default='n2n')
parser.add_argument("--imgPath", help="directory in which test images are", default="test_images/AST/raw/")
parser.add_argument("--savePath", help="where to save denoised images", default="denoised/")

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# check args
args = parser.parse_args()
img_type = args.imgType
model_name = args.modelName
test_img_path = args.imgPath
save_path = args.savePath
if not test_img_path.endswith('/'):
    test_img_path = test_img_path+'/'
if not save_path.endswith('/'):
    save_path = save_path+'/'
if not path.exists(save_path):
    os.mkdir(save_path)

# import all dependencies
import numpy as np
from tifffile import imread, imsave

import torch
from csbdeep.models import CARE
from n2v.models import N2V
from nets.dncnn import DnCNN
from nets.resnet import ResNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load test images
def cut_off(img_, max_value):
    img = np.copy(img_)
    img = img.astype('float32')
    mask = img > max_value
    img[mask] = max_value
    img = img / max_value * 1.0
    return img

test_imgs = []
files = os.listdir(test_img_path)
for file in files:
    print('loading ' + file)
    image = imread(test_img_path + file)
    max_value = np.percentile(image, 99.5)
    test_imgs.append(cut_off(image, max_value))
test_imgs = np.stack(test_imgs)
print('done')

# load nets
model = None
if model_name == 'care':
    print('loading CARE.')
    model = CARE(config=None, name='care', basedir='nets/models_{}/'.format(img_type))
if model_name == 'n2n':
    print('loading N2N')
    model = CARE(config=None, name='n2n', basedir='nets/models_{}/'.format(img_type))
if model_name == 'n2v':
    print('loading N2V')
    model = N2V(config=None, name='n2v', basedir='nets/models_{}/'.format(img_type))
if model_name == 'dncnn':
    print('loading DnCNN.')
    model = DnCNN(bias=False)
    model.load_state_dict(torch.load('nets/models_{}/dncnn/dncnn.pt'.format(img_type), map_location=device))
    model.eval()
if model_name == 'resnet':
    print('loading ResNet.')
    model = ResNet()
    model.load_state_dict(torch.load('nets/models_{}/resnet/resnet.pt'.format(img_type), map_location=device))
    model.eval()

# denoising and save denoised images
if model_name in ['care', 'n2n', 'n2v']:
    imgs = test_imgs[..., np.newaxis]
    preds = model.keras_model.predict(imgs)[...,0]
if model_name in ['resnet', 'bf-dncnn']:
    imgs = np.expand_dims(test_imgs, axis=1)
    imgs = torch.as_tensor(imgs.astype('float32'), device=device)
    preds = model(imgs).detach().numpy()[:, 0, ...]

for i, file in enumerate(files):
    imsave(save_path+file, preds[i])
    print("image {} saved to {}".format(file, save_path))
print("done")
