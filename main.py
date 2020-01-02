import os

# first we need to download the libs
try:
	os.system('pip3 install -r requirements.txt')
except:
	print("Check your Python3 and Pip installations.")

# import libs
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# create a gym env
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# check for a gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
