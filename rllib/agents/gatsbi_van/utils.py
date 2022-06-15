from ray.rllib.utils.framework import try_import_torch
import numpy as np
from torch.distributions import constraints
from torch.distributions import gamma

from ray.rllib.agents.gatsbi_van.baselines.clear_objects_config import CFG
from ray.rllib.agents.gatsbi_van.baselines.reward_utils import tensor_tolerance
from ray.rllib.agents.gatsbi_van.modules.arch import ARCH

import math
import enum
torch, nn = try_import_torch()

class Axis(enum.Enum):
    """Maps axes to image indices, assuming that 0th dimension is the batch,
      and the 1st dimension is the channel."""
    y = 2
    x = 3

# Custom initialization for different types of layers
def scale_action(raw_action):
    """
        Scale the raw action output from the policy to fit the 
        action space of the agent.
        This scaled action should not be parsed to the environment.
    """
    if 'bair_push' in ARCH.REAL_WORLD and raw_action.shape[-1] == ARCH.ACTION_DIM:
        return raw_action
    if torch.is_tensor(raw_action):
        a_pos = raw_action[..., :3].clone()
        axis = raw_action[..., 3:6].clone()
        theta = raw_action[..., 6].clone()[..., None]

        # scale action elements
        theta = CFG.THETA * theta
        a_pos = a_pos / (torch.norm(a_pos, dim=-1)[..., None] + 1e-6)
        a_pos = a_pos * CFG.ACTION_SCALE

        scaled_action = torch.cat([a_pos, axis, theta], dim=-1)
    else:
        a_pos = raw_action[..., :3]
        axis = raw_action[..., 3:6]
        theta = raw_action[..., 6][..., None]

        # scale action elements
        theta *= CFG.THETA
        a_pos = a_pos / (np.linalg.norm(a_pos, axis=-1)[..., None] + 1e-6)
        a_pos *= CFG.ACTION_SCALE

        scaled_action = np.concatenate([a_pos, axis, theta], axis=-1)
    return scaled_action

def generate_general_gaussian(position, loc, scale=0.5, power=8.0):
    """
        Implementation of generalized normal distribution in Pytorch.
        # TODO (chmin): just exponentiate this!
        ref: https://github.com/tensorflow/probability/blob/
        f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/
        python/distributions/generalized_normal.py#L183

        position (position of datapoints): [1, K, 1 (64), 64 (1)]
            former is x axis, latter is y axis
            K is usually 1.
        loc (center of gaussian): [B, K, 1, 1]

    """
    scale_tensor = torch.tensor(scale).to(position.device)
    power_tensor = torch.tensor(power).to(position.device)
    one = torch.tensor(1.0, dtype=torch.float32).to(position.device)
    two = torch.tensor(2.0, dtype=torch.float32).to(position.device)

    log_normalization = (torch.log(two) + torch.log(scale_tensor) + 
        torch.lgamma(one + torch.reciprocal(power_tensor))) # []

    log_unnormalized = - torch.pow(torch.abs(position - loc) / 
        (scale_tensor + 1e-6), power_tensor) # [B, K, 1 (64), 64 (1)]

    log_pdf = log_unnormalized - log_normalization # [B, K, 1 (64), 64 (1)]
    pdf = log_pdf.exp() # [B, K, 1 (64), 64 (1)]
    return pdf

def _get_pixel_grid(axis, width):
    """Returns an array of length `width` containing pixel coordinates."""
    if axis == Axis.x: # pixel width
        return torch.linspace(-1.0, 1.0, width)  # Left is negative, right is positive.
    elif axis == Axis.y: # pixel height
        return torch.linspace(1.0, -1.0, width)  # Top is positive, bottom is negative.

def generate_heatmaps(keypoints, sigma=2.0, heatmap_width=64):
    """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).

    Args:
    keypoints: [batch_size, K=num_keypoints, 3] tensor of keypoints where the 1st
        dimension contains (x, y, scale) triplets.
    sigma: Std. dev. of the Gaussian blob, in units of heatmap pixels.
    heatmap_width: Width of output heatmaps in pixels.

    Returns:
    A [batch_size, num_keypoints, heatmap_width, heatmap_width] tensor.
    """
    with torch.no_grad():
        coordinates, map_scales = torch.split(keypoints, [2, 1], dim=-1) # [B, K, 2], [B, K, 1]
        # split into two dim and one dim
        def get_grid(axis):
            grid = _get_pixel_grid(axis, heatmap_width)
            shape = [1, 1, 1, 1]
            shape[axis.value] = -1
            return torch.reshape(grid, shape)

        # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
        # TODO(cheolhui): determine the axes for # keypoints and the features (x,y,scale) 
        x_coordinates = coordinates[:, :, None, None, 0] # [B, K, 1, 1] #! mean
        y_coordinates = coordinates[:, :, None, None, 1] # [B, K, 1, 1] #! mean

        # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
        sigma = torch.tensor(sigma).float()
        keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

        x_vec = generate_general_gaussian(position=get_grid(Axis.x).cuda(device=keypoints.device),
            loc=x_coordinates, scale=1.0) # [B, K, 1, 64]
        y_vec = generate_general_gaussian(position=get_grid(Axis.y).cuda(device=keypoints.device),
            loc=y_coordinates, scale=1.0) # [B, K, 64, 1]

        # x_vec =  (- (get_grid(Axis.x).cuda(device=keypoints.device) - x_coordinates).pow(2) / keypoint_width).exp() # [B, K, 1, W]
        # y_vec =  (- (get_grid(Axis.y).cuda(device=keypoints.device) - y_coordinates).pow(2) / keypoint_width).exp() # [B, K, H, 1]
        maps =  x_vec * y_vec # [B, K, H, W]
        maps = maps / maps.max()
    return maps

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class GRUCell(nn.GRUCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)


# Custom Tanh Bijector due to big gradients through Dreamer Actor
class TanhBijector(torch.distributions.Transform):

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self):
        super().__init__()

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def sign(self):
        return 1.

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where((torch.abs(y) <= 1.),
                        torch.clamp(y, -0.99999997, 0.99999997), y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - nn.functional.softplus(-2. * x))



# Modified from https://github.com/juliusfrost/dreamer-pytorch
# refer to the Concept of context manager.  
class FreezeParameters:
    def __init__(self, parameters):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i]
