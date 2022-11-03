import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


def run_softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.threshold = None
        
    def fit(self, X, y):
        softX = run_softmax(X, axis=1)
        best_thresh = 0.0
        best_acc = 0
        for threshold in np.linspace(0, 1.0, self.n_trials + 2):
            preds = np.apply_along_axis(lambda row: int(max(row) < threshold), axis=1, arr=softX)
            accuracy = accuracy_score(y, preds)
            if accuracy > best_acc:
                best_acc = accuracy
                best_thresh = threshold
                # print(f"New best threshold: {best_thresh} ({round(best_acc,3) * 100}%)")
        self.threshold = best_thresh

    def predict(self, X):
        softX = run_softmax(X, axis=1)
        preds = np.apply_along_axis(lambda row: int(max(row) < self.threshold), axis=1, arr=softX)
        return preds