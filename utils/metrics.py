# Owned by Johns Hopkins University, created prior to 5/28/2020
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def accuracy(preds, target, one_hot_labels):
    if one_hot_labels:
        return torch.tensor(accuracy_score(preds.cpu().numpy().argmax(1), target.cpu().numpy().argmax(1)))
    return torch.tensor(accuracy_score(preds.cpu().numpy().argmax(1), target.cpu().numpy()))


def macro_f1(preds, target, one_hot_labels):
    if one_hot_labels:
        return torch.tensor(f1_score(preds.cpu().argmax(1), target.cpu().numpy().argmax(1), average='macro'))
    return torch.tensor(f1_score(preds.cpu().argmax(1), target.cpu().numpy(), average='macro'))


def genus_accuracy(preds, gtarget, sptarget):
    gacc = accuracy_score(preds[0].cpu().numpy().argmax(1), gtarget.cpu().numpy().argmax(1))
    return torch.tensor(gacc)


def species_accuracy(preds, gtarget, sptarget):
    spacc = accuracy_score(preds[1].cpu().numpy().argmax(1), sptarget.cpu().numpy().argmax(1))
    return torch.tensor(spacc)


def genus_f1_score(preds, gtarget, sptarget):
    return torch.tensor(f1_score(preds[0].cpu().argmax(1), gtarget.cpu().numpy().argmax(1), average='macro'))


def species_f1_score(preds, gtarget, sptarget):
    return torch.tensor(f1_score(preds[1].cpu().argmax(1), sptarget.cpu().numpy().argmax(1), average='macro'))


def f1_score_slow(y_true, y_pred, threshold=0.5):
    """
    Usage: f1_score(py_true, py_pred)
    """
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean((precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))
