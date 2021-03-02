import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", help="choose the model for training", default="n2v")
parser.add_argument("--dataDir", help="dir of training data", default="data/")
parser.add_argument("--numImgsInTif", help="how many images in one .tif file you want to add to training set", default=1)
parser.add_argument("--expandData", help="whether to expand data by averaging 2,4,8,16 raw images", default=True)
parser.add_argument("--shape", help="the shape of input patches", default=(256,256))
parser.add_argument("--testSize", help="the size of validation set", default=0.1)
parser.add_argument("--augment", help="whether to do data augmentation", default=True)
parser.add_argument("--batchSize", help="training batch size", default=16)
parser.add_argument("--epochs", help="training epochs", default=100)
parser.add_argument("--lr", help="training learning rate", default=0.0004)
parser.add_argument("--modelName", help="name of the model to be saved", default="n2v")
parser.add_argument("--baseDir", help="the dir to save the model", default="save_models")
parser.add_argument("--bias", help="whether to add bias to networks", default=False)
parser.add_argument("--structN2VMask", help="blind mask for structure Noise2Void", default=None)


# check and print arguments
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
if not args.dataDir.endswith('/'):
    args.dataDir = args.dataDir + '/'
if not args.baseDir.endswith('/'):
    args.baseDir = args.baseDir + '/'
print(args)


# import dependencies
from training.train_care import train_care
from training.train_n2v import train_n2v
from training.train_n2n import train_n2n
from training.train_dncnn import train_dncnn
from training.train_resnet import train_resnet
from training.train_unet import train_unet


# create configuration dictionary
config_dict = {'data_dir': args.dataDir, 'num_imgs_in_tif': args.numImgsInTif,
               'expand_data': args.expandData, 'patch_shape': args.shape,
               'test_size': args.testSize, 'augment': args.augment,
               'batch_size': args.batchsSize, 'epochs': args.epochs,
               'lr': args.lr, 'model_name': args.modelName, 'basedir': args.baseDir}


# train and save model
model_type = args.model
if model_type == 'care':
    train_care(**config_dict)
if model_type == 'n2v':
    train_n2v(**config_dict)
if model_type == 'n2v':
    config_dict['structN2Vmask'] = args.structN2VMask
    train_n2n(**config_dict)
if model_type == 'dncnn':
    config_dict['bias'] = args.bias
    train_dncnn(**config_dict)
if model_type == 'resnet':
    train_resnet(**config_dict)
if model_type == 'unet':
    config_dict['bias'] = args.bias
    train_unet(**config_dict)
