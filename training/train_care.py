import numpy as np
from csbdeep.models import Config, CARE
from preprocessing import extract_data, generate_patches_from_list


def train_care(data_dir, num_imgs_in_tif,
               expand_data, patch_shape=(256,256),
               test_size=0.1, augment=True, unet_depth=2,
               batch_size=32, epochs=100, lr=0.0004,
               model_name='care', basedir='save_models'):
    raw_x, raw_y = extract_data(data_dir=data_dir, num_imgs_in_tif=num_imgs_in_tif,
                                expand_data=expand_data, data_type='XY')
    print(raw_x.shape)
    split_idx = int(raw_x.shape[0]*(1-test_size))
    X_train = generate_patches_from_list([raw_x[:split_idx]], shape=patch_shape, augment=augment)[...,np.newaxis]
    X_val = generate_patches_from_list([raw_x[split_idx:]], shape=patch_shape, augment=augment)[...,np.newaxis]
    Y_train = generate_patches_from_list([raw_y[:split_idx]], shape=patch_shape, augment=augment)[...,np.newaxis]
    Y_val = generate_patches_from_list([raw_y[split_idx:]], shape=patch_shape, augment=augment)[...,np.newaxis]
    if X_train.shape[0] < batch_size:
        raise Exception("The number of training data is smaller than batch size. Try to set batch size smaller.")
    np.random.seed(0)
    np.random.shuffle(X_train)
    np.random.seed(0)
    np.random.shuffle(Y_train)
    config = Config(axes='SYXC', n_channel_in=1, n_channel_out=1,
                    unet_kern_size=3, train_batch_size=batch_size,
                    train_steps_per_epoch=int(X_train.shape[0]/batch_size),
                    allow_new_parameters=True, train_epochs=epochs,
                    train_learning_rate=lr, unet_n_depth=unet_depth)
    model = CARE(config=config, name=model_name, basedir=basedir)
    model.train(X_train, Y_train, validation_data=(X_val, Y_val))
    return
