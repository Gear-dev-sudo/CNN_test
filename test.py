 
import numpy as np
import torch
import models.vgg as vgg
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import argparse
import measures
from torchvision import transforms, datasets

print( torch.cuda.is_available())
