import numpy as np
import os
from tifffile import imread

import torch
from torch.utils.data import TensorDataset, DataLoader


def cut_off(img, max_value):
    img = img.astype('float32')
    mask = img > max_value
    img[mask] = max_value
    img = img / max_value * 1.0
    return img


def extract_data(data_dir, num_imgs_in_tif=1, expand_data=False, data_type='XY'):
    '''

    :param data_dir: image directory
    :param num_imgs_in_tif: number of images to extract from one .tif file
    :param expand_data: expand data by average 2,4,8,16 raw images as training data, default=False
    :param data_type: 'XY', 'XX', or 'X', default='XY'
    :return: training data
    '''

    raw_x = []
    raw_y = []

    data_files = os.listdir(data_dir)
    for file in data_files:
        if file.endswith('.tif'):
            img_series = imread(data_dir+file)
            num_img = img_series.shape[0]
            max_value = np.percentile(img_series, 99.5)
            new_img_series = np.zeros(img_series.shape)
            for i in range(img_series.shape[0]):
                new_img_series[i] = cut_off(img_series[i], max_value)
            gt = np.mean(new_img_series, axis=0)
            if expand_data is False:
                raw_x.append(new_img_series[:num_imgs_in_tif])
                if data_type == 'XY':
                    for i in range(num_imgs_in_tif):
                        raw_y.append(gt)
                if data_type == 'XX':
                    raw_y.append(new_img_series[int(num_img/2):int(num_img/2)+num_imgs_in_tif])
            else:
                avg_num = [1, 2, 4, 8, 16]
                for s in avg_num:
                    for n in range(num_imgs_in_tif):
                        avg_img1 = np.mean(new_img_series[n:(n + s)], axis=0)
                        if data_type == 'XX':
                            avg_img2 = np.mean(new_img_series[(n+int(num_img/2)):(n+int(num_img/2)+s)], axis=0)
                            raw_y.append(avg_img2)
                        raw_x.append(avg_img1)
                if data_type == 'XY':
                    for i in range(5*num_imgs_in_tif):
                        raw_y.append(gt)
            del img_series
    if expand_data is True:
        raw_x = np.stack(raw_x)
    else:
        raw_x = np.concatenate(raw_x, axis=0)
    raw_y = np.stack(raw_y)
    if data_type == 'XY' or 'XX':
        return raw_x, raw_y
    if data_type == 'X':
        return raw_x


def extract_patches(data, shape):
    patches = []
    if data.shape[1] > shape[0] and data.shape[2] > shape[1]:
        for y in range(0, data.shape[1] - shape[0] + 1, shape[0]):
            for x in range(0, data.shape[2] - shape[1] + 1, shape[1]):
                patches.append(data[:, y:y + shape[0], x:x + shape[1]])
        return np.concatenate(patches)
    elif data.shape[1] == shape[0] and data.shape[2] == shape[1]:
        return data
    else:
        print("'shape' is too big.")


def augment_patches(patches):
    augmented = np.concatenate((patches, np.rot90(patches, k=1, axes=(1, 2))))
    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented


def generate_patches(data, shape=(256,256), augment=True):
    patches = extract_patches(data, shape=shape)
    if shape[-2] == shape[-1]:
        if augment is True:
            patches = augment_patches(patches)
    else:
        if augment:
            print("XY-Plane is not square. Omit augmentation!")
    print('Generated patches:', patches.shape)
    return patches


def generate_patches_from_list(data, shape=(256, 256), augment=True):
    patches = []
    for img in data:
        for s in range(img.shape[0]):
            p = generate_patches(img[s][np.newaxis], shape=shape, augment=augment)
            patches.append(p)
    patches = np.concatenate(patches, axis=0)
    return patches


def load_data(data_dir, num_imgs_in_tif, expand_data, test_size, shape, batch_size, augment, device):
    raw_x, raw_y = extract_data(data_dir=data_dir,
                                num_imgs_in_tif=num_imgs_in_tif,
                                expand_data=expand_data,
                                data_type='XY')
    split_idx = int(raw_x.shape[0]*(1-test_size))
    X_train = generate_patches_from_list([raw_x[:split_idx]], shape=shape, augment=augment)[:,np.newaxis,...]
    X_val = generate_patches_from_list([raw_x[split_idx:]], shape=shape, augment=augment)[:,np.newaxis,...]
    Y_train = generate_patches_from_list([raw_y[:split_idx]], shape=shape, augment=augment)[:,np.newaxis,...]
    Y_val = generate_patches_from_list([raw_y[split_idx:]], shape=shape, augment=augment)[:,np.newaxis,...]
    np.random.seed(0)
    np.random.shuffle(X_train)
    np.random.seed(0)
    np.random.shuffle(Y_train)
    print(X_train.shape, X_val.shape)
    del raw_x
    del raw_y
    dataset_sizes = dict()
    dataset_sizes['training'] = X_train.shape[0]
    dataset_sizes['val'] = X_val.shape[0]
    X_train = torch.as_tensor(X_train, device=device)
    X_val = torch.as_tensor(X_val, device=device)
    Y_train = torch.as_tensor(Y_train, device=device)
    Y_val = torch.as_tensor(Y_val, device=device)
    dataset1 = TensorDataset(X_train, Y_train)
    dataset2 = TensorDataset(X_val, Y_val)
    loader = dict()
    loader['training'] = DataLoader(dataset1, batch_size=batch_size)
    loader['val'] = DataLoader(dataset2, batch_size=batch_size)

    return dataset_sizes, loader
