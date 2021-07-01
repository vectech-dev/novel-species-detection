# Owned by Johns Hopkins University, created prior to 5/28/2020
"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init


from torchvision import models
from pretrainedmodels.models import xception
# from .xceptionalt import xceptionalt

class TwoXception(nn.Module):
    def __init__(self, config, num_classes=2):
        super(TwoXception, self).__init__()
        self.genus = config.genus

        if config.pretrained:
            print('Loading pretrained weights...')
            self.xception_base = xception(pretrained="imagenet")
        else:
            self.xception_base = xception(pretrained=None)

        # Remove the last layer
        self.xception_base = torch.nn.Sequential(*(list(self.xception_base.children())[:-1]))

        if config.genus:
            print("Using two output Xception with {} classes".format(num_classes))
            self.species_fc = nn.Linear(in_features=2048, out_features=num_classes[0], bias=True)
            self.genus_fc = nn.Linear(in_features=2048, out_features=num_classes[1], bias=True)

        else:
            print("Using Xception with {} classes".format(num_classes))
            self.species_fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, X):
        X = self.xception_base(X)

        X = nn.ReLU(inplace=True)(X)
        X = F.adaptive_avg_pool2d(X, (1, 1))
        X = X.view(X.size(0), -1)
        
        sp = self.species_fc(X)

        if self.genus:
            gn = self.genus_fc(X)
            return sp, gn

        return sp

def Mos_Xception(config, model_name = "xception", num_classes=2):
    if model_name == "xception":
        # model = TwoXception(config, num_classes)

        ## Uncomment if having problems using Xception
        print("Using Xception with {} classes".format(num_classes))
        if config.pretrained:
            print('Loading pretrained weights...')
            model = xception(pretrained="imagenet")
        else:
            model = xception(pretrained=None)
        
        model.last_linear = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        ## Old Xception definition end

    if model_name == "xceptionconcat":
        print("Using XceptionConcat with {} classes".format(num_classes))
        if pretrained:
            print('Loading pretrained weights...')
            model = xception_concat(pretrained="imagenet")
        else:
            model = xception_concat(pretrained=None)
        
        model.last_linear = nn.Linear(in_features=5320, out_features=num_classes, bias=True)

    # if model_name == "xceptionalt":
    #     print("Using XceptionAlt with {} classes".format(num_classes))
    #     if pretrained:
    #         print('Loading pretrained weights...')
    #         model = xceptionalt(pretrained=True)
    #     else:
    #         model = xceptionalt(pretrained=False)
        
    #     model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    return model




__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class XceptionConcat(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionConcat, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x1 = self.conv1(input)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x1 = self.conv2(x1)
        x2 = self.bn2(x1.clone())
        x2 = self.relu2(x2)

        x2 = self.block1(x2)
        x2 = self.block2(x2)
        x2 = self.block3(x2)
        x3 = self.block4(x2.clone())
        x3 = self.block5(x3)
        x3 = self.block6(x3)
        x4 = self.block7(x3.clone())
        x4 = self.block8(x4)
        x4 = self.block9(x4)
        x5 = self.block10(x4.clone())
        x5 = self.block11(x5)
        x5 = self.block12(x5)

        x6 = self.conv3(x5.clone())
        x6 = self.bn3(x6)
        x6 = self.relu3(x6)

        x6 = self.conv4(x6)
        x6 = self.bn4(x6)
        return x1, x2, x3, x4, x5, x6

    # def logits(self, features):
    #     x = nn.ReLU(inplace=True)(features)

    #     x = F.adaptive_avg_pool2d(x, (1, 1))
    #     x = x.view(x.size(0), -1)
    #     x = self.last_linear(x)
    #     return x

    def forward(self, input):
        x1, x2, x3, x4, x5, x6 = self.features(input)
        
        x1 = nn.ReLU(inplace=True)(x1)
        x2 = nn.ReLU(inplace=True)(x2)
        x3 = nn.ReLU(inplace=True)(x3)
        x4 = nn.ReLU(inplace=True)(x4)
        x5 = nn.ReLU(inplace=True)(x5)
        x6 = nn.ReLU(inplace=True)(x6)

        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x3 = F.adaptive_avg_pool2d(x3, (1, 1))
        x4 = F.adaptive_avg_pool2d(x4, (1, 1))
        x5 = F.adaptive_avg_pool2d(x5, (1, 1))
        x6 = F.adaptive_avg_pool2d(x6, (1, 1))
        
        x_cat = torch.cat([x1,x2,x3,x4,x5,x6], dim=1)

        x_cat = x_cat.view(x_cat.size(0), -1)
        x_cat = self.last_linear(x_cat)

        return x_cat


def xception_concat(num_classes=1000, pretrained='imagenet'):
    model = XceptionConcat(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = XceptionConcat(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model