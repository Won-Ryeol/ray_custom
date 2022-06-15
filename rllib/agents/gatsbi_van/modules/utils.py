import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import pickle
import os
import numpy as np
import torch
from torch import nn
import numbers
import time
from .arch import ARCH

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TensorAccumulator:
    """ concat tensors with optional (right) padding if shapes don't match """
    def __init__(self, pad=False, pad_value=0):
        self.items = {}
        self.pad = pad
        self.pad_value = pad_value

    def add(self, key, value):
        def _get_item_with_padding(item, item_shape, max_shape):
            diff = max_shape - item_shape
            zeros = torch.zeros_like(diff)

            padding = reversed([(x, y) for x, y, in zip(zeros, diff)])
            padding = [y for x in padding for y in x]
            return F.pad(item, padding, value=self.pad_value)

        if key not in self.items:
            self.items[key] = value.clone()
        else:
            prev_items = self.items[key]
            new_item = value.clone()
            if self.pad:
                prev_shape = torch.tensor(self.items[key].shape[1:])
                new_shape = torch.tensor(value.shape[1:])
                max_shape = torch.max(prev_shape, new_shape)

                if not torch.all(prev_shape == max_shape):
                    prev_items = _get_item_with_padding(prev_items, prev_shape, max_shape)

                if not torch.all(new_shape == max_shape):
                    new_item = _get_item_with_padding(new_item, new_shape, max_shape)

            self.items[key] = torch.cat([prev_items, new_item], dim=0)

    def get(self, key, default=None):
        return self.items.get(key, default)
class Timer:
    def __init__(self):
        from collections import OrderedDict
        self.start = time.perf_counter()
        self.times = OrderedDict()
    def check(self, name):
        self.times[name] = time.perf_counter() - self.start
        self.start = time.perf_counter()

class SmoothedValue:
    """
    Record the last several values, and return summaries
    """
    
    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0
    
    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value
    
    @property
    def median(self):
        return np.median(np.array(self.values))
    
    @property
    def avg(self):
        return np.mean(self.values)
    
    @property
    def global_avg(self):
        return self.sum / self.count


class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)
    
    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, item):
        self.values[key].update(item)

def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch: [2, 3] -> [B*N, 2, 3]
    theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1) #! like tf.tile
    # set scaling -> [[s, 0], [0, s]]
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9) # [B*N] 'h'
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9) # [B*N] 'w'
    
    # set translation @ the last column [..., -1]
    theta[:, 0, -1] = z_where[:, 2] if not inverse else - z_where[:, 2] / (z_where[:, 0] + 1e-9) # 'x'
    theta[:, 1, -1] = z_where[:, 3] if not inverse else - z_where[:, 3] / (z_where[:, 1] + 1e-9) # 'y'
    # 2. construct sampling grid: In grid.permute(0, 3, 1, 2)[:, 1:] grid.permute(0, 3, 1, 2)[:, :1]
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=True) # generates a 2D flow field (sampling grid), given a batch of affine matrices theta.
    # 3. sample image from grid of shape [B*N, 3, H, W] grid[0][..., 0][None] grid[0][..., 1][None]
    return F.grid_sample(image, grid)

def transform_tensors(x, func):
    """
    Transform each tensor in x using func. We preserve the structure of x.
    Args:
        x: some Python objects
        func: function object

    Returns:
        x: transformed version
    """
    # make recursion to preserve the structure of 'x'
    if isinstance(x, torch.Tensor):
        return func(x)
    elif isinstance(x, numbers.Number):
        return x
    elif isinstance(x, list):
        return [transform_tensors(item, func) for item in x]
    elif isinstance(x, dict):
        return {k: transform_tensors(v, func) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(transform_tensors(item, func) for item in x)
    else:
        raise TypeError('Non tensor or number object must be either tuple, list or dict, '
                        'but {} encountered.'.format(type(x)))


class Checkpointer:
    def __init__(self, path, max_num=3):
        self.max_num = max_num
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.listfile = os.path.join(path, 'model_list.pkl')
        if not os.path.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)
    
    def save(self, model, optimizer, lr_scheduler, epoch, global_step):
        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'epoch': epoch,
            # 'iteration': iteration,
            'global_step': global_step
        }
        filename = os.path.join(self.path, 'model_{:09}.pth'.format(global_step + 1))
        
        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if os.path.exists(model_list[0]):
                    # if int(model_list[0][-13:-4]) != ARCH.MODULE_TRAINING_SCHEME[0]:
                    #     os.remove(model_list[0])
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(filename)
            model_list = list(set(model_list))
            model_list.sort()
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
        
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            print(f'Checkpoint has been saved to "{filename}".')
            
    def save_to_path(self, model, optimizer, epoch, global_step, path):
        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'lr_scheduler': lr_scheduler.state_dict() if optimizer else None,
            'epoch': epoch,
            # 'iteration': iteration,
            'global_step': global_step
        }
        with open(path, 'wb') as f:
            torch.save(checkpoint, f)
            # print(f'Checkpoint has been saved to "{filename}".')
    
    def load(self, path, model, optimizer, lr_scheduler=None):
        """
        Return starting epoch
        """
        
        if path == '':
            with open(self.listfile, 'rb') as f:
                model_list = pickle.load(f)
                if len(model_list) == 0:
                    print('No checkpoint found. Starting from scratch')
                    return None
                else:
                    if ARCH.PTH_START:
                        step = str(ARCH.PTH_START)
                        f_name = 'model_' + '0' * (9 -len(step)) + step + '.pth'
                        want_dir = os.path.join(cfg.checkpointdir, cfg.exp_name, f_name)
                        path = want_dir
                    else:
                        path = model_list[-1]
        
        assert os.path.exists(path), f'Checkpoint {path} does not exist.'
        if model_list[-1] != path:
            model_list.append(path)
        
        print('Loading checkpoint from {}...'.format(path))
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint.pop('model'))
        if optimizer:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint.pop('lr_scheduler'))
        print('Checkpoint loaded.')
        return checkpoint

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

