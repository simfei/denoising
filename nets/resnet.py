import torch.nn as nn
from torchvision.models import resnet50


def ResNet():
    model = resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    model = nn.Sequential(*list(model.children())[:-2])
    model.add_module('upsampling1',
                     nn.ConvTranspose2d(2048, 512, kernel_size=(2, 2), stride=(2, 2))
                     )
    model.add_module('bn1', nn.BatchNorm2d(512))
    model.add_module('upsampling2',
                     nn.ConvTranspose2d(512, 128, kernel_size=(2, 2), stride=(2, 2))
                     )
    model.add_module('bn2', nn.BatchNorm2d(128))
    model.add_module('upsampling3',
                     nn.ConvTranspose2d(128, 32, kernel_size=(2, 2), stride=(2, 2))
                     )
    model.add_module('bn3', nn.BatchNorm2d(32))
    model.add_module('upsampling4',
                      nn.ConvTranspose2d(32, 1, kernel_size=(2, 2), stride=(2, 2))
                     )
    return model
