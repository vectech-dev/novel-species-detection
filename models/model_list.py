# Owned by Johns Hopkins University, created prior to 5/28/2020
import sys
from functools import partial
import torch.nn as nn
import torchvision
import pretrainedmodels
# from models.densenet import Mos_DenseNet
# from models.uselessnet import UselessNet
# from models.resnet import Mos_ResNet
# from models.inception import Mos_Inception
from models.xception import Mos_Xception
# from models.xceptionmod import Mos_XceptionMod
# from models.senet import Mos_SENet
# from models.sononet import Mos_Sononet_Attn
# from models.xception_attention import Mos_Xception_Attn
# from models.pnasnet import Mos_PNASnet
# from models.Jongchan.bamnet import Mos_BAMNet
# from models.dilatedrn import Mos_DRN
# from models.airx import Mos_AirX
# from models.shake_pyramidnet import Mos_Shake_Pyramid
# from models.shake_shake import Mos_Shake_Shake
# from models.nts_net import Mos_NTSNet
# from models.nofe_head import NofEHead
# from models import mxresnet as mxresnetorig
# from pytorchcv.model_provider import get_model as ptcv_get_model
# from efficientnet_pytorch import EfficientNet
# from models.mranresnet import MRANNet


# def resnet18(config, num_classes=2):
#     return Mos_ResNet(config=config, modeln="resnet18", num_classes=num_classes)

# def resnet34(config, num_classes=2):
#     return Mos_ResNet(config=config, modeln="resnet34", num_classes=num_classes)

# def resnet50(config, num_classes=2):
#     return Mos_ResNet(config=config, modeln="resnet50", num_classes=num_classes)

# def resnet101(config, num_classes=2):
#     return Mos_ResNet(config=config, modeln="resnet101", num_classes=num_classes)

# def resnet152(config, num_classes=2):
#     return Mos_ResNet(config=config, modeln="resnet152", num_classes=num_classes)

# def nofe_xcpetion(config):
#     x = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
#     xc = nn.Sequential(*[i for i in x.children()][:-1])
#     nofehead = NofEHead(config, in_channels=2048)
#     nofe = nn.Sequential(xc, nofehead)
#     return nofe

# def nofe_mxresnet(base, config):
#     x = base()
#     xc = nn.Sequential(*[i for i in x.children()][:-3])
#     nofehead = NofEHead(config, in_channels=2048)
#     nofe = nn.Sequential(xc, nofehead)
#     return nofe

# def resattnet92(config):
#     net = ptcv_get_model("resattnet92", pretrained=False)
#     net.output = nn.Linear(net.output.in_features, config.num_species.sum(), bias=(net.output.bias is not None))
#     return net

# def efficientnet_b6(config):
#     if config.pretrained:
#         net = EfficientNet.from_pretrained("efficientnet-b6")
#     else:
#         net = EfficientNet.from_name('efficientnet-b6')
#     net._fc = nn.Linear(net._fc.in_features, config.num_species.sum(), bias=(net._fc.bias is not None))
#     return net

# def mobilenet(config):
#     net = torchvision.models.mobilenet_v2(pretrained=config.pretrained)
#     net.classifier[1] = nn.Linear(net.classifier[1].in_features, config.num_species.sum(), bias=(net.classifier[1].bias is not None))
#     return net

# def shufflenet(config):
#     net = torchvision.models.shufflenet_v2_x1_0(pretrained=config.pretrained)
#     net.fc = nn.Linear(net.fc.in_features, config.num_species.sum(), bias=(net.fc.bias is not None))
#     return net

# def shufflenetbig(config):
#     net = torchvision.models.shufflenet_v2_x2_0(pretrained=False)
#     net.fc = nn.Linear(net.fc.in_features, config.num_species.sum(), bias=(net.fc.bias is not None))
#     return net

# me = sys.modules[__name__]
# for d in [18, 34, 50, 101, 152]:
#     name = f'nofe_mxresnet{d}'
#     base = getattr(mxresnetorig, name.replace('nofe_', ''))
#     setattr(me, name, partial(nofe_mxresnet, base=base))

# def mxresnet(base, config):
#     return base(c_out=config.num_species.sum())

# me = sys.modules[__name__]
# for d in [18, 34, 50, 101, 152]:
#     name = f'mxresnet{d}'
#     base = getattr(mxresnetorig, name)
#     setattr(me, name, partial(mxresnet, base=base))


# def densenet121(pretrained=False, drop_rate=0.):
#     return Mos_DenseNet(modeln="densenet121", pretrained=pretrained, 
#             drop_rate=drop_rate)

# def densenet169(pretrained=False, drop_rate=0.):
#     return Mos_DenseNet(modeln="densenet169", pretrained=pretrained, 
#             drop_rate=drop_rate)

# def densenet201(pretrained=False, drop_rate=0.):
#     return Mos_DenseNet(modeln="densenet201", pretrained=pretrained, 
#             drop_rate=drop_rate)

# def densenet161(pretrained=False, drop_rate=0.):
#     return Mos_DenseNet(modeln="densenet161", pretrained=pretrained, 
#             drop_rate=drop_rate)

# def bninception(config):
#     return Mos_Inception(config=config, model_name = 'bninception', num_classes=config.num_species.sum())

# def inceptionv2(config):
#     return Mos_Inception(config=config, model_name = 'inceptionv2', num_classes=config.num_species.sum())

# def inceptionv4(config):
#     return Mos_Inception(config=config, model_name = 'inceptionv4', num_classes=config.num_species.sum())

# def inceptionresnetv2(config):
#     return Mos_Inception(config=config, model_name = 'inceptionresnetv2', num_classes=config.num_species.sum())

def xception(config):
    return Mos_Xception(config=config, model_name = 'xception', num_classes=config.num_species.sum())

# def xceptionmod(config):
#     return Mos_XceptionMod(config=config, model_name = 'xceptionmod', num_classes=config.num_species.sum())

# def xceptionconcat(config):
#     return Mos_Xception(config=config, model_name = 'xceptionconcat', num_classes=config.num_species.sum())

# def xceptionalt(config):
#     return Mos_Xception(config=config, model_name = 'xceptionalt', num_classes=config.num_species.sum())

# def mranresnet(config):
#     return MRANNet(config=config, num_classes = config.num_species.sum())

# def seinceptionv3(pretrained=False, drop_rate=0.):
#     return Mos_SENet(model_name = 'seinceptionv3', pretrained=pretrained, 
#                                 drop_rate=drop_rate)

# def sononet_grid_attention(pretrained=False, drop_rate=0.):
#     return Mos_Sononet_Attn(model_name = "sononet_grid_attention", pretrained=False, 
#                                 drop_rate=0.)

# def xception_grid_attention(pretrained=False, drop_rate=0.):
#     return Mos_Xception_Attn(model_name = "xception_grid_attention", pretrained=False, 
#                                 drop_rate=0.)

# def pnasnet(pretrained=False, drop_rate=0.):
#     return Mos_PNASnet(model_name = "pnasnet", pretrained=False, drop_rate=0.)

# def resnet18cbam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet18cbam", drop_rate=0., pretrained=False)

# def resnet18bam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet18bam", drop_rate=0., pretrained=False)

# def resnet34cbam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet34cbam", drop_rate=0., pretrained=False)

# def resnet34bam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet34bam", drop_rate=0., pretrained=False)

# def resnet50cbam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet50cbam", drop_rate=0., pretrained=False)

# def resnet50bam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet50bam", drop_rate=0., pretrained=False)

# def resnet101cbam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet101cbam", drop_rate=0., pretrained=False)

# def resnet101bam(pretrained=False, drop_rate=0):
#     return Mos_BAMNet(model_name = "resnet101bam", drop_rate=0., pretrained=False)

# def drnd54(pretrained=False, drop_rate=0):
#     return Mos_DRN(model_name = "drn-d-54", drop_rate=0., pretrained=True)

# def drnd105(pretrained=False, drop_rate=0):
#     return Mos_DRN(model_name = "drn-d-105", drop_rate=0., pretrained=True)

# def airx50(pretrained=False, drop_rate=0):
#     return Mos_AirX(model_name = "airx50_32x4d", drop_rate=0., pretrained=True)

# def airx101(pretrained=False, drop_rate=0):
#     return Mos_AirX(model_name = "airx101_32x4d_r2", drop_rate=0., pretrained=True)

# def shakepyramid(config):
#     return Mos_Shake_Pyramid(config=config, num_classes = num_classes)

# def shakeres20(config):
#     return Mos_Shake_Shake(model_name = "shakeres20", num_classes=num_classes)

# def shakeres26(config):
#     return Mos_Shake_Shake(config=config, model_name = "shakeres26", num_classes=num_classes)

# def shakerescustom(config):
#     return Mos_Shake_Shake(config=config, model_name = "shakerescustom", num_classes=num_classes)

# def ntsnet(config):
#     return Mos_NTSNet(config=config, model_name = "ntsnet", num_classes=num_classes)

# def uselessnet():
#     return UselessNet()