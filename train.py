import sys
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", help="choose the model for training: "
                                    "care, dncnn, resnet, n2n, n2v, or pn2v", default="n2v")
parser.add_argument("--dataDir", help="dir of training data", default="data/")
parser.add_argument("--numImgsInTif", help="how many frames in one .tif file you want to add to training set", default=1)
parser.add_argument("--expandData", help="expand data by averaging 2, 4, 8, 16 raw images", default='True')
parser.add_argument("--shape", help="the shape of input patches, e.g. input 256, the shape is (256,256)", type=int, default=256)
parser.add_argument("--testSize", help="the size of validation set", type=float, default=0.1)
parser.add_argument("--augment", help="do data augmentation", default='True')
parser.add_argument("--batchSize", help="training batch size", type=int, default=16)
parser.add_argument("--epochs", help="training epochs", type=int, default=100)
parser.add_argument("--lr", help="training learning rate", type=float, default=0.0004)
parser.add_argument("--modelName", help="name of the trained model to be saved. "
                                        "For PN2V, modelName is also the path to save the model. "
                                        "For structured N2V, add 'struct' as prefix.", default="n2v")
parser.add_argument("--baseDir", help="the base directory to save the model", default="save_models")
parser.add_argument("--bias", help="whether to add bias to networks, only used for DnCNN and UNet", default='False')
parser.add_argument("--structN2VMask", help="blind mask for structured Noise2Void, "
                                            "choose from 1x3, 1x5, 1x9, 3x1, 5x1, 9x1", default=None)
parser.add_argument("--unetDepth", help="depth of UNet, only used for model: care, n2n, n2v, and pn2v",
                    type=int, default=2)

# check and print arguments
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

blind_masks_choices = ['1x3', '1x5', '1x9', '3x1', '5x1', '9x1']
blind_masks = {'1x3': [[0,0,0,0,1,1,1,0,0,0,0]],
               '1x5': [[0,0,0,1,1,1,1,1,0,0,0]],
               '1x9': [[0,1,1,1,1,1,1,1,1,1,0]],
               '3x1': [[0],[0],[0],[0],[1],[1],[1],[0],[0],[0],[0]],
               '5x1': [[0],[0],[0],[1],[1],[1],[1],[1],[0],[0],[0]],
               '9x1': [[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]]}
args = parser.parse_args()
if args.expandData == 'True':
    expand_data = True
else:
    expand_data = False
if args.augment == 'True':
    augment = True
else:
    augment = False
if args.bias == 'True':
    use_bias = True
else:
    use_bias = False
if not args.dataDir.endswith('/'):
    args.dataDir = args.dataDir + '/'
if not args.baseDir.endswith('/'):
    args.baseDir = args.baseDir + '/'
if args.model == 'pn2v':
    if not os.path.exists(args.modelName):
        os.mkdir(args.baseDir+args.modelName)
if not os.path.exists(args.baseDir):
    os.mkdir(args.baseDir)
if args.structN2VMask is not None:
    if args.structN2VMask not in blind_masks_choices:
        raise Exception("Choices for blind masks are within 1x3, 1x5, 1x9, 3x1, 5x1 and 9x1.")
    else:
        structN2Vmask = blind_masks[args.structN2VMask]
else:
    structN2Vmask = None
print(args)


# import dependencies
sys.path.append("training")
from train_care import train_care
from train_n2v import train_n2v
from train_n2n import train_n2n
from train_dncnn import train_dncnn
from train_resnet import train_resnet
from train_unet import train_unet
from train_pn2v import train_pn2v


# create configuration dictionary
config_dict = {'data_dir': args.dataDir, 'num_imgs_in_tif': args.numImgsInTif,
               'expand_data': expand_data, 'patch_shape': (args.shape, args.shape),
               'test_size': args.testSize, 'augment': augment,
               'batch_size': args.batchSize, 'epochs': args.epochs, 'lr': args.lr,
               'model_name': args.modelName, 'basedir': args.baseDir}
if args.model == 'pn2v':
    config_dict['patch_shape'] = args.shape

# train and save model
model_type = args.model
if model_type == 'care':
    config_dict['unet_depth'] = args.unetDepth
    train_care(**config_dict)
if model_type == 'n2n':
    config_dict['unet_depth'] = args.unetDepth
    train_n2n(**config_dict)
if model_type == 'n2v' or model_type.startswith('struct_n2v'):
    config_dict['unet_depth'] = args.unetDepth
    config_dict['structN2Vmask'] = structN2Vmask
    train_n2v(**config_dict)
if model_type == 'dncnn':
    config_dict['bias'] = use_bias
    train_dncnn(**config_dict)
if model_type == 'resnet':
    train_resnet(**config_dict)
if model_type == 'unet':
    config_dict['bias'] = use_bias
    train_unet(**config_dict)
if model_type == 'pn2v':
    config_dict['unet_depth'] = args.unetDepth
    train_pn2v(**config_dict)
