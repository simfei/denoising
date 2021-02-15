import numpy as np
from preprocessing import extract_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V


def get_patches(imgs, shape=(256,256), augment=True):
    datagen = N2V_DataGenerator()
    patches = datagen.generate_patches_from_list([imgs[...,np.newaxis]], shape=shape, augment=augment)
    return patches


def train_n2v(data_dir, num_imgs_in_tif, expand_data, test_size=0.1,
              patch_shape=(256,256), augment=True,
              batch_size=128, epochs=200, lr=0.0004,
              model_name='n2v', basedir='save_models'
             ):
    raw_x = extract_data(data_dir=data_dir, num_imgs_in_tif=num_imgs_in_tif,
                         expand_data=expand_data, data_type='X')
    split_idx = int(raw_x.shape[0]*(1-test_size))
    X_train = get_patches(raw_x[:split_idx], shape=patch_shape, augment=augment)
    X_val = get_patches(raw_x[split_idx:], shape=patch_shape, augment=augment)
    config = N2VConfig(X_train, unet_kern_size=3,
                       use_residual=True, unet_n_depth=2,
                       unet_n_first=32, batch_norm=True,
                       train_steps_per_epoch=int(X_train.shape[0]/128),
                       train_epochs=epochs, train_loss='mse',
                       train_learning_rate=lr,
                       train_batch_size=batch_size, n2v_perc_pix=1.5,
                       n2v_patch_shape=(64, 64),
                       n2v_manipulator='uniform_withCP',
                       n2v_neighborhood_radius=5)
    model = N2V(config=config, name=model_name, basedir=basedir)
    model.train(X_train, X_val)
