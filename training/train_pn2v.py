import numpy as np
import torch
from nets.pn2v import histNoiseModel
from nets.pn2v import training
from nets.unet.model import UNet
from preprocessing import extract_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_pn2v(data_dir, num_imgs_in_tif=1,
               expand_data=False, patch_shape=100,
               test_size=0.1, augment=True,
               batch_size=16, epochs=200, lr=0.0004,
               model_name='pn2v/', basedir='save_models/',
               unet_depth=3
               ):
    if not model_name.endswith('/'):
        model_name = model_name + '/'
    if not basedir.endswith('/'):
        basedir = basedir + '/'
    raw_x, observation, signal = extract_data(data_dir=data_dir, num_imgs_in_tif=num_imgs_in_tif,
                                              expand_data=expand_data, data_type='X', noise_model=True)

    maxVal = 1
    minVal = 0
    bins = 256
    histogram = histNoiseModel.createHistogram(bins, minVal, maxVal, observation, signal)
    np.save(basedir+model_name+'noiseModel.npy', histogram)
    noiseModel = histNoiseModel.NoiseModel(histogram, device=device)

    split_idx = int(raw_x.shape[0] * (1 - test_size))
    my_train_data = raw_x[:split_idx].copy()
    np.random.shuffle(my_train_data)
    my_val_data = raw_x[split_idx:].copy()
    np.random.shuffle(my_val_data)
    if my_train_data.shape[0] < batch_size:
        raise Exception("The number of training data is smaller than batch size. Try to set batch size smaller.")

    net = UNet(800, depth=unet_depth)
    training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                          postfix='PN2V', directory=basedir+model_name,
                          patchSize=patch_shape, numMaskedPixels=int(patch_shape*patch_shape/32.0),
                          noiseModel=noiseModel, device=device,
                          numOfEpochs=epochs,
                          stepsPerEpoch=int(my_train_data.shape[0] / batch_size),
                          virtualBatchSize=20,
                          batchSize=batch_size,
                          learningRate=lr,
                          augment=augment)
    return

