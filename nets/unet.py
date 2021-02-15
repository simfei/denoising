import torch
import torch.nn as nn


def conv_block(channel_in, channel_mid, channel_out, batch_norm, bias):
    layers = []
    layers.append(nn.Conv2d(channel_in, channel_mid, 3, bias=bias, padding=1))
    if batch_norm:
        layers.append(nn.BatchNorm2d(channel_mid, affine=bias))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(channel_mid, channel_out, 3, bias=bias, padding=1))
    if batch_norm:
        layers.append(nn.BatchNorm2d(channel_out, affine=bias))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, input_size=256, channel_in=1, channel_out=1, batch_norm=True, bias=True,
                 use_risidual=True, last_activation='linear'):
        super().__init__()
        self.use_risidual = use_risidual
        self.batch_norm = batch_norm
        if last_activation is None:
            self.last_activation = 'relu'
        else:
            self.last_activation = last_activation


        # down
        self.conv_down1 = conv_block(channel_in, 32, 32, batch_norm, bias)
        self.pool1 = nn.MaxPool2d(2)

        self.conv_down2 = conv_block(32, 64, 64, batch_norm, bias)
        self.pool2 = nn.MaxPool2d(2)

        # middle
        self.conv_middle = conv_block(64, 128, 64, batch_norm, bias)

        # up
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_up1 = conv_block(128, 64, 32, batch_norm, bias)

        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_up2 = conv_block(64, 32, 16, batch_norm, bias)

        self.final = nn.Conv2d(16, channel_out, 3, bias=bias, padding=1)
        self.final_act = nn.Linear(input_size, input_size, bias=bias)

        if self.last_activation == 'relu':
            self.last_act = nn.ReLU(inplace=True)
        else:
            self.last_act = nn.Linear(input_size, input_size, bias=bias)

    def forward(self, x):
        layer = x
        skip_layers = []

        # down
        layer = self.conv_down1(layer)
        skip_layers.append(layer)
        layer = self.pool1(layer)

        layer = self.conv_down2(layer)
        skip_layers.append(layer)
        layer = self.pool2(layer)

        # middle
        layer = self.conv_middle(layer)

        # up
        layer = self.upsampling1(layer)
        concat1 = torch.cat((layer, skip_layers[1]), dim=1)
        layer = self.conv_up1(concat1)

        layer = self.upsampling2(layer)
        concat2 = torch.cat((layer, skip_layers[0]), dim=1)
        layer = self.conv_up2(concat2)

        layer = self.final(layer)
        layer = self.final_act(layer)

        if self.use_risidual is True:
            layer = torch.add(layer, x)
            layer = self.last_act(layer)

        return layer
