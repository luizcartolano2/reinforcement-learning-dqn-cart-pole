# import libs
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(env, screen_width):
    """
        Return the cart location on the image.

        :input: screen_width - size of the screen
        :input: env - a gym enviroment

        :return: a int with the middle of the cart

    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env, device):
    """
        Returned screen requested by gym is 400x600x3, but is sometimes larger,
        such as 800x1200x3. Transpose it into torch order (CHW).
        
        :input: env - a gym enviroment
        :input: device - tells if use cuda or cpu

        :return: 
    """
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


if __name__ == '__main__':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # check for a gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a gym env
    env = gym.make('CartPole-v0').unwrapped

    env.reset()
    plt.figure()
    plt.imshow(get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()

    import time; time.sleep(15)

    env.close()
