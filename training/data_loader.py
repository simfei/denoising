import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import extract_data, generate_patches_from_list


def load_data(data_dir, num_imgs_in_tif, expand_data, test_size, shape, batch_size, augment, device):
    # load data for torch implementation
    raw_x, raw_y = extract_data(data_dir=data_dir,
                                num_imgs_in_tif=num_imgs_in_tif,
                                expand_data=expand_data,
                                data_type='XY')
    split_idx = int(raw_x.shape[0]*(1-test_size))
    X_train = generate_patches_from_list([raw_x[:split_idx]], shape=shape, augment=augment)[:,np.newaxis,...]
    X_val = generate_patches_from_list([raw_x[split_idx:]], shape=shape, augment=augment)[:,np.newaxis,...]
    Y_train = generate_patches_from_list([raw_y[:split_idx]], shape=shape, augment=augment)[:,np.newaxis,...]
    Y_val = generate_patches_from_list([raw_y[split_idx:]], shape=shape, augment=augment)[:,np.newaxis,...]
    if X_train.shape[0] < batch_size:
        raise Exception("The number of training data is smaller than batch size. Try to set batch size smaller.")
    np.random.seed(0)
    np.random.shuffle(X_train)
    np.random.seed(0)
    np.random.shuffle(Y_train)
    # print(X_train.shape, X_val.shape)
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