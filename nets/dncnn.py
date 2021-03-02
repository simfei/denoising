import torch.nn as nn
import torch.nn.init as init


class DnCNN(nn.Module):
    # reference: https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch
    def __init__(self, depth=17, n_channels=64, image_channels=1,
                 use_bnorm=True, kernel_size=3, bias=False):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []

        layers.append(nn.Conv2d(image_channels, n_channels,
                      kernel_size=kernel_size, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels,
                          kernel_size=kernel_size, padding=padding, bias=bias))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, affine=bias))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels,
                      kernel_size=kernel_size, padding=padding, bias=bias))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        # out: residual, y: noisy input
        return y - out

