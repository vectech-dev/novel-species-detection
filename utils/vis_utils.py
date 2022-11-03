# Owned by Johns Hopkins University, created prior to 5/28/2020
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plttensor(imgt, mean=torch.tensor([0.485, 0.456, 0.406]),
				 std=torch.tensor([0.229, 0.224, 0.225]), vis=True, getim=False):
	imgt = imgt.detach().cpu()
	img = imgt.permute(1,2,0) * std + mean
	img = img.detach().cpu().numpy();
	if vis:
		plt.imshow(img)
		plt.show()
	if getim:
		return img




