import numpy as np
import os
from tifffile import imread


def cut_off(img_, max_value):
    img = np.copy(img_)
    img = img.astype('float32')
    mask = img > max_value
    img[mask] = max_value
    img = img / max_value * 1.0
    return img


def extract_data(data_dir, num_imgs_in_tif=1, expand_data=False, data_type='XY', noise_model=False):
    '''

    :param data_dir: image directory
    :param num_imgs_in_tif: number of images to extract from one .tif file
    :param expand_data: expand data by average 2,4,8,16 raw images as training data, default=False
    :param data_type: 'XY', 'XX', or 'X', default='XY'. 'XY' for extracting noisy inputs and clean targets,
                      'XX' for extracting noisy inputs and noisy targets, and 'X' for extracting noisy inputs.
    :param noise_model: only used for training PN2V. Set True if noise model is estimated by using training data,
                        else set False, default=False. When set True, data_type must be 'X'.
    :return: training data
    '''

    observation = None
    signal = None
    if noise_model is True:
        observation = []
        signal = []
        if data_type is not 'X':
            raise Exception('data_type {} is not compatible with noise_model True!'.format(data_type))

    raw_x = []
    raw_y = []

    data_files = os.listdir(data_dir)
    for file in data_files:
        if file.endswith('.tif'):
            img_series = imread(data_dir+file)
            num_img = img_series.shape[0]
            max_value = np.percentile(img_series, 99.5)
            new_img_series = cut_off(img_series, max_value)
            gt = np.mean(new_img_series, axis=0)
            if expand_data is False:
                raw_x.append(new_img_series[:num_imgs_in_tif])
                if data_type == 'XY':
                    for i in range(num_imgs_in_tif):
                        raw_y.append(gt)
                if data_type == 'XX':
                    raw_y.append(new_img_series[int(num_img/2):(int(num_img/2)+num_imgs_in_tif)])
                if noise_model is True:
                    observation.append(new_img_series)
                    signal.append(gt)
            else:
                avg_num = [1, 2, 4, 8, 16]
                for s in avg_num:
                    for n in range(num_imgs_in_tif):
                        avg_img1 = np.mean(new_img_series[n:(n + s)], axis=0)
                        raw_x.append(avg_img1)
                        if noise_model is True:
                            observation.append(avg_img1)
                            signal.append(gt)
                        if data_type == 'XX':
                            avg_img2 = np.mean(new_img_series[(n+int(num_img/2)):(n+int(num_img/2)+s)], axis=0)
                            raw_y.append(avg_img2)
                if data_type == 'XY':
                    for i in range(5*num_imgs_in_tif):
                        raw_y.append(gt)
            del img_series
    if expand_data is True:
        raw_x = np.stack(raw_x)
        if noise_model is True:
            observation = np.stack(observation)
    else:
        raw_x = np.concatenate(raw_x, axis=0)
        if noise_model is True:
            observation = np.concatenate(observation, axis=0)
    if noise_model is True:
        signal = np.stack(signal)
    if data_type is not 'X':
        raw_y = np.stack(raw_y)
    if data_type in ['XY', 'XX']:
        return raw_x, raw_y
    if data_type == 'X':
        if noise_model is True:
            return raw_x, observation, signal
        else:
            return raw_x


def extract_patches(data, shape):
    # reference: https://github.com/juglab/n2v/blob/master/n2v/internals/N2V_DataGenerator.py
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
    # reference: https://github.com/juglab/n2v/blob/master/n2v/internals/N2V_DataGenerator.py
    augmented = np.concatenate((patches,
                                np.rot90(patches, k=1, axes=(1, 2)),
                                np.rot90(patches, k=2, axes=(1, 2)),
                                np.rot90(patches, k=3, axes=(1, 2))
                                ))
    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented


def generate_patches(data, shape=(256,256), augment=True):
    # reference: https://github.com/juglab/n2v/blob/master/n2v/internals/N2V_DataGenerator.py
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
    # reference: https://github.com/juglab/n2v/blob/master/n2v/internals/N2V_DataGenerator.py
    patches = []
    for img in data:
        for s in range(img.shape[0]):
            p = generate_patches(img[s][np.newaxis], shape=shape, augment=augment)
            patches.append(p)
    patches = np.concatenate(patches, axis=0)
    return patches

