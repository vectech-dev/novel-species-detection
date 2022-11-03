# Owned by Johns Hopkins University, created prior to 5/28/2020
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from IPython.core.debugger import set_trace

class MosLoss(nn.Module):
    def __init__(self, config):
        super(MosLoss, self).__init__()
        self.config = config
        self.loss_dict = copy.deepcopy(config.loss_dict)
        assert len(self.loss_dict) > 0
        self.lweights = []
        self.mag_scale = []
        for k,v in self.loss_dict.items():
            assert 'weight' in v
            assert 'mag_scale' in v
            self.lweights.append(v['weight'])
            self.mag_scale.append(v['mag_scale'])
            del v['weight']
            del v['mag_scale']
        self.lweights = torch.tensor(self.lweights).cuda()
        self.mag_scale = torch.tensor(self.mag_scale).cuda()
        assert self.lweights.sum() == 1

        self.losses = [globals()[k](**v) for k,v in self.loss_dict.items()]

    def forward(self, inputs, targets, reduction=None):
        if len(targets.shape) != len(inputs.shape):
            if self.config.genus:
                raise NotImplementedError
            else:
                temp = torch.zeros((targets.shape[0], self.config.num_species.sum()),
                             device=targets.device)
                temp.scatter_(1, targets.view(-1, 1), 1)
                targets = temp
        loss = self.losses[0](inputs, targets)*self.mag_scale[0]
        # print("l0:", loss)
        for i in range(1, len(self.losses)):
            l_i = self.losses[i](inputs, targets)*self.mag_scale[i]
            # print("l{}:".format(i), l_i)
            loss += l_i
        # print("Weights: ", self.lweights)
        loss = (loss*self.lweights).sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        print("Focal Loss with gamma = ", gamma)
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

class FocalLoss2(nn.Module):
    def __init__(self, gamma=2, genus_weight=0.5):
        super().__init__()
        print("Focal Loss 2 Stream with gamma = ", gamma)
        self.fl1 = FocalLoss(gamma = gamma)
        self.fl2 = FocalLoss(gamma = gamma)
        assert genus_weight>=0 and genus_weight<=1
        self.genus_weight = genus_weight

    def forward(self, input, target_genus, target_species):
        loss_genus = self.fl1(input[0], target_genus)
        loss_species = self.fl2(input[1], target_species)
        total_loss = self.genus_weight*loss_genus + (1-self.genus_weight)*loss_species
        return total_loss

class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        print("F1 Loss")

    def forward(self, input, target):
        tp = (target*input).sum(0)
        # tn = ((1-target)*(1-input)).sum(0)
        fp = ((1-target)*input).sum(0)
        fn = (target*(1-input)).sum(0)

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)

        f1 = 2*p*r / (p+r+1e-9)
        f1[f1!=f1] = 0.
        return 1 - f1.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("Dice Loss")

    def forward(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2.*intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalTverskyLoss(nn.Module):
    """
    https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    Focal Tversky Loss. Tversky loss is a special case with gamma = 1
    """
    def __init__(self, alpha = 0.4, gamma = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        print("Focal Tversky Loss with alpha = ", alpha, ", gamma = ", gamma)

    def tversky(self, input, target):
        smooth = 1.
        input = torch.sigmoid(input)

        target_pos = target.view(-1)
        input_pos = input.view(-1)
        true_pos = (target_pos * input_pos).sum()
        false_neg = (target_pos * (1-input_pos)).sum()
        false_pos = ((1-target_pos)*input_pos).sum()
        return (true_pos + smooth)/(true_pos + self.alpha*false_neg + \
                        (1-self.alpha)*false_pos + smooth)

    def forward(self, input, target):
        pt_1 = self.tversky(input, target)
        return (1-pt_1).pow(self.gamma)


class MRANLoss(nn.Module):
    def __init__(self, config):
        super(MRANLoss, self).__init__()
        self.config = config
        self.mosloss = MosLoss(config)
        
    def forward(self, inputs, targets):
        feats, cmmd_loss = inputs
        class_loss = self.mosloss(feats, targets)
        return class_loss + self.config.cmmd_loss_weight * cmmd_loss.mean()
