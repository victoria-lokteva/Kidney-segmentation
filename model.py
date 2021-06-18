import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

"""transfer learning with FCN"""


def prepare_model(pretrained=False):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)
    if pretrained:
        frozen_layers = nn.Sequential(*list(model.children()))[:-1]
        for parameter in frozen_layers.parameters():
            parameter.requires_grad = False
    # we have 2 clases: is a glomeruli's pixel or not
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    return model
