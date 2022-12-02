import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


def num_param():
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  scale_factor = 12
  DCE_net = model.enhance_net_nopool(scale_factor).cuda()
  DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
  count = sum(p.numel() for p in DCE_net.parameters() if p.requires_grad)
  print("num of parameters: ",count)


if __name__ == '__main__':
  with torch.no_grad():
    num_param()
