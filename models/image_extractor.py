import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock
from functools import partial
from .vision_transformer import vit_base, vit_small


def get_image_extractor(arch='resnet18', pretrained=True):
    if 'resnet' in arch:
        resnet = Get_resnet(arch=arch, pretrained=pretrained)
        return resnet
    elif 'vit' in arch:
        vit = Get_vit(arch=arch, pretrained=pretrained)
        return vit
    else:
        raise NotImplementedError


class Get_resnet(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True):
        super(Get_resnet, self).__init__()

        if arch == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            # model.fc = nn.Sequential()
            # self.model = model
            modules = list(model.children())[:-2]  # 返回resnet子模块
            self.model = nn.Sequential(*modules)
        elif arch == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Sequential()
            self.model = model

    def forward(self, x):
        return self.model(x)


class Get_vit(nn.Module):
    def __init__(self, arch='vit-base', pretrained=True):
        super(Get_vit, self).__init__()
        if arch == 'vit-base':
            model = vit_base()
            if pretrained:
                state_dict = torch.load('./models/dino_vitbase16_pretrain.pth')
                model.load_state_dict(state_dict, strict=True)
            self.model = model
        if arch == 'vit-small':
            model = vit_small()
            if pretrained:
                state_dict = torch.load('./models/dino_deitsmall16_pretrain.pth')
                model.load_state_dict(state_dict, strict=True)
            self.model = model

    def forward(self, x):
        return self.model(x)[:, 0, :]


class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
