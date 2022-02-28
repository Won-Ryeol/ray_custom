import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from .arch import ARCH
from contextlib import nullcontext
import enum

EPSILON = 1e-6  # Constant for numerical stability.

# TODO (cheolhui): modify all tensorflow ops into that of pytorch.
# TODO (cheolhui): adjust the indentation level of codes.




def get_heatmap_penalty(weight_matrix, factor):
    """L1-loss on mean heatmap activations, to encourage sparsity.
    args:
      weight_matrix: heatmap output of shape [B, 3 + 2, G, G]
    """
    weight_shape = weight_matrix.size()
    assert len(weight_shape) == 4, weight_shape
    heatmap_mean = torch.mean(weight_matrix, dim=(2, 3)) # [B, 3+2]
    penalty = torch.mean(torch.abs(heatmap_mean), dim=1) # [B,]
    return penalty * factor

def maps_to_keypoints(heatmaps):
    """Turns feature-detector heatmaps into (x, y, scale) keypoints.

    This function takes a tensor of feature maps as input. Each map is normalized
    to a probability distribution and the location of the mean of the distribution
    (in image coordinates) is computed. This location is used as a low-dimensional
    representation of the heatmap (i.e. a keypoint).

    To model keypoint presence/absence, the mean intensity of each feature map is
    also computed, so that each keypoint is represented by an (x, y, scale)
    triplet.

    Args:
      heatmaps: [batch_size, K, H, W] tensors. -> num_keypoints are determined from config.
    Returns:
      A [batch_size, num_keypoints, 3] tensor with (x, y, scale)-triplets for each
      keypoint. Coordinate range is [-1, 1] for x and y, and [0, 1] for scale.
    """

    # Check that maps are non-negative:
    map_min = torch.min(heatmaps) # scalar, minimum among all pixels
    
    x_coordinates = _maps_to_coordinates(heatmaps, Axis.x) # [B, K] -> x coords for k keypoints.
    y_coordinates = _maps_to_coordinates(heatmaps, Axis.y) # Axis.y (H) = 2
    map_scales = torch.mean(heatmaps, dim=[2, 3]) # [B, K, G, G] -> [B, K] mean over H, W dim

    # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
    # degeneracy between the encoder and decoder heatmap scales and ensures that
    # the scales are in a reasonable range for the RNN:
    map_scales /= (EPSILON + torch.max(map_scales, dim=-1, keepdim=True)[0])# [B, K] -> [B, K]
    if len(x_coordinates.shape) == 1:
        x_coordinates = x_coordinates[None]
        y_coordinates = y_coordinates[None]
    return torch.stack([x_coordinates, y_coordinates, map_scales], dim=-1) # [B, K, 3]

def _maps_to_coordinates(maps, axis):
    """Reduces heatmaps to coordinates along one axis (x or y).

    Args:
      maps: heatmap of size [batch_size, K, H, W] tensors. 
      axis: Axis Enum.

    Returns:
      A [batch_size, num_keypoints, 2] tensor with (x, y)-coordinates.
    """
    width = maps.size()[axis.value] # get the size of y; Height (x : Width) of the feature map
    grid = _get_pixel_grid(axis, width) # [W, ]
    grid = grid.cuda(device=maps.device)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1 # if x (width): 3rd else y (height): 2nd
    grid = torch.reshape(grid, shape) # [1, 1, H (1), 1 (W)]
    
    if axis == Axis.x:
      marginalize_dim = 2 # if get x position (W) of keypoints, marginalize over y (H).
    elif axis == Axis.y:
      marginalize_dim = 3

    # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
    # if keepdim is False, should have been [B, K, H (1), 1 (W)]
    weights = torch.sum(maps + EPSILON, dim=marginalize_dim, keepdim=True) 
    # sum over remaining dim.
    weights /= torch.sum(weights, dim=axis.value, keepdim=True)  # [B, K, 1, 1]
    # Compute the center of mass of the marginalized maps to obtain scalar
    # coordinates: [B, 3 + 2, 1, 1]
    coordinates = torch.sum(weights * grid, dim=axis.value, keepdim=True) # [B, K, 1, 1]
    return torch.squeeze(coordinates) # [B, 3+2]


def keypoints_to_maps(keypoints, sigma=1.0, heatmap_width=16):
    """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).

    Args:
      keypoints: [batch_size, K=num_keypoints, 3] tensor of keypoints where the 1st
        dimension contains (x, y, scale) triplets.
      sigma: Std. dev. of the Gaussian blob, in units of heatmap pixels.
      heatmap_width: Width of output heatmaps in pixels.

    Returns:
      A [batch_size, num_keypoints, heatmap_width, heatmap_width] tensor.
    """

    coordinates, map_scales = torch.split(keypoints, [2, 1], dim=-1) # [B, K, 2], [B, K, 1]
    # split into two dim and one dim
    def get_grid(axis):
        grid = _get_pixel_grid(axis, heatmap_width)
        shape = [1, 1, 1, 1]
        shape[axis.value] = -1
        return torch.reshape(grid, shape)

    # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
    # TODO(cheolhui): determine the axes for # keypoints and the features (x,y,scale) 
    x_coordinates = coordinates[:, :, None, None, 0] # [B, K, 1, 1]
    y_coordinates = coordinates[:, :, None, None, 1] # [B, K, 1, 1]
    # test1 = get_grid(Axis.x).cuda(device=keypoints.device) -> expand x_coordinates 
    # test2 = get_grid(Axis.y).cuda(device=keypoints.device)

    # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
    sigma = torch.tensor(sigma).float()
    keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0
    x_vec =  (- (get_grid(Axis.x).cuda(device=keypoints.device) - x_coordinates).pow(2) / keypoint_width).exp() # [B, K, 1, W]
    y_vec =  (- (get_grid(Axis.y).cuda(device=keypoints.device) - y_coordinates).pow(2) / keypoint_width).exp() # [B, K, H, 1]
    maps =  x_vec * y_vec # [B, K, H, W]
    return maps * map_scales[:, :, 0, None, None] # [B, K, H, W]


def _get_pixel_grid(axis, width):
    """Returns an array of length `width` containing pixel coordinates."""
    if axis == Axis.x: # pixel width
        return torch.linspace(-1.0, 1.0, width)  # Left is negative, right is positive.
    elif axis == Axis.y: # pixel height
        return torch.linspace(1.0, -1.0, width)  # Top is positive, bottom is negative.


def add_coord_channels(img):
    """Adds channels containing pixel indices (x and y coordinates) to an image.

    Note: This has nothing to do with keypoint coordinates. It is just a data
    augmentation to allow convolutional networks to learn non-translation-
    equivariant outputs. This is similar to the "CoordConv" layers:
    https://arxiv.org/abs/1603.09382.

    Args:
    image_tensor: [batch_size, H, W, C] tensor.

    Returns:
    [batch_size, H, W, C + 2] tensor with x and y coordinate channels.
    """
    batch_size, C, y_size, x_size = img.size()
    
    x_grid = torch.linspace(-1.0, 1.0, x_size, device=img.device) # [B, ]
    x_map = x_grid[None, None, None, :].repeat(batch_size, 1, y_size, 1) # [B, C=1, H, 1]

    y_grid = torch.linspace(1.0, -1.0, y_size, device=img.device) # [B, ] 
    y_map = y_grid[None, None, :, None].repeat(batch_size, 1, 1, x_size) # [B, C=1, 1, W]
    # TODO (cheolhui): check their shapes
    return torch.cat([img, x_map, y_map], dim=1) # concat along channel dim [B, C + 2, H, W]


def temporal_separation_loss(coords):
    """Encourages keypoint to have different temporal trajectories.

    If two keypoints move along trajectories that are identical up to a time-
    invariant translation (offset), this suggest that they both represent the same
    object and are redundant, which we want to avoid.
    # TODO (cheolhui): note that we rather require this feature!

    To measure this similarity of trajectories, we first center each trajectory by
    subtracting its mean. Then, we compute the pairwise distance between all
    trajectories at each timepoint. These distances are higher for trajectories
    that are less similar. To compute the loss, the distances are transformed by
    a Gaussian and averaged across time and across trajectories.

    Args:
      coords: [B, T, num_landmarks, 3] coordinate tensor.

    Returns:
      Separation loss.
    """
    x = coords[..., 0] # [B, T, K] -> 1st element of keypoint predictions
    y = coords[..., 1] # [B, T, K] -> 2nd element of keypoint predictions

    # Center of keypoints in the trajectories difference to the mean along T axis
    # [B, T, K] - [B, 1, K] = [B, T, K]
    x = x - torch.mean(x, dim=1, keepdims=True) # [B, 1, K]
    y = y - torch.mean(y, dim=1, keepdims=True) # [B, 1, K]

    # Compute pairwise distance matrix d:
    # [B, T, K, 1] - [B, T, 1, K] =  [B, T, K, K]
    d = ((x[:, :, :, None] - x[:, :, None, :]) ** 2.0 +
         (y[:, :, :, None] - y[:, :, None, :]) ** 2.0)    

    # Temporal mean:
    d = torch.mean(d, dim=1) # [B, K, K]

    # Apply Gaussian function such that loss falls off with distance -> larger distance -> lower rloss
    loss_matrix = (-d / (2.0 * ARCH.KYPT_SEP_LOSS_SIGMA ** 2.0)).exp() # [B, K, K]
    loss = torch.sum(loss_matrix, dim=(1, 2))  # Sum over all pairs.
    
    # Subtract sum of values on diagonal, which are always 1 -> since exp(0) == 1
    loss = loss - ARCH.NUM_KEYPOINTS

    # Normalize by maximal possible value. The loss is now scaled between 0 (all
    # keypoints are infinitely far apart) and 1 (all keypoints are at the same
    # location):
    loss = loss / ARCH.NUM_KEYPOINTS * (ARCH.NUM_KEYPOINTS - 1)

    return loss

class Axis(enum.Enum):
    """Maps axes to image indices, assuming that 0th dimension is the batch,
      and the 1st dimension is the channel."""
    y = 2
    x = 3

class Interpolate(nn.Module):
    def __init__(self, size, mode):
      super(Interpolate, self).__init__()
      self.interp = F.interpolate
      self.size = size
      self.mode = mode
    
    def forward(self, x):
      x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)

class KeypointModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.img_to_kypt = ImgToKyptNet()
        self.kypt_to_img = KyptToImgNet()
        self.kypt_dyna = KyptDynaNet()
        # placeholder Tensor for the first frame of episode.
        self.first_frame = torch.ones((1, 3, 64, 64))
        self.first_kypt = torch.ones([3, ARCH.NUM_KEYPOINTS])

    def anneal(self, global_step):
        self.global_step = global_step
    
    def predict_keypoints(self, agent_seq, state=None, global_step=0, leverage=False,
            backup=False, model_T=0):
        """
        From bg_things['agent'], we infer the keypoints on the robot's end-effector.
        The model input: agent_seq
                output: reconstructed_images, observed_keypoints, predicted_keypoints
        args:
            agent_seq: A seq of agent layer embedded in the background. -> m_agent * c_agent
        """
        # --- Vision model ---
        # 1) get keypoint from raw observation.
        # In: agent-seq [B, T, 3, 64, 64]
        # Out: obs_kypt [B, T, 3, K]
        _, T, *_ = agent_seq.size()

        if leverage:
            detached_timesteps = T - model_T
        else:
            detached_timesteps = 0

        # Out - obs_kypts [B, T, K, 3], heatmap_reg_losses [B, T]
        observed_keypoints, _, heatmap_reg_losses = self.img_to_kypt(agent_seq, leverage=leverage,
            model_T=model_T)  # Out - kedefypoints, heatmaps
        heatmap_reg_losses = torch.sum(heatmap_reg_losses[:, detached_timesteps:], dim=1)

        # 2) reconstruct images from keypoints.
        # Out - [B, T, 3, 64, 64]
        reconstructed_images, gaussian_maps = self.kypt_to_img(keypoints=observed_keypoints,  # [B, T, K, 3]
            first_frame=agent_seq[:, 0, ...],  # [B, 3, H, W]
            first_frame_keypoints=observed_keypoints[:, 0, ...],
            leverage=leverage,
            model_T=model_T)  # [B, K, 3]
        
        #  --- Dynamics model ---
        observed_keypoints_detach = observed_keypoints.detach()
        # In - [B, T, K, 3], Out - [B, T, K, 3] & [B, T]
        predicted_keypoints, kl_divergence = self.kypt_dyna.encode(observed_keypoints_detach, 
            state, leverage, backup, model_T)

        # 3) --- Losses ---
        # backprop for agent_seq
        # 3-1) reconstruction loss

        image_loss = nn.MSELoss(reduction='none')(agent_seq, reconstructed_images) / 2.0
        image_loss = torch.sum(image_loss[:, detached_timesteps:], dim=(1, 2, 3, 4))  # [B, ]
        # normalize by batch_size and sequence_length:
        image_loss = image_loss / T
        # 3-2) seperation loss : penalize if keypoints' trajectories overlap
        # In: [B, T, K, 3] & out: scalar
        separation_loss = ARCH.KYPT_SEP_LOSS_SCALE * temporal_separation_loss(
            observed_keypoints[:, detached_timesteps:])
        vrnn_coord_pred_loss = nn.MSELoss(reduction='none')(observed_keypoints_detach, predicted_keypoints) / 2.0
        vrnn_coord_pred_loss = torch.sum(vrnn_coord_pred_loss[:, detached_timesteps:], dim=(1, 2, 3))  # [B, ]
        # Normalize by batch_size and sequence length:
        vrnn_coord_pred_loss = vrnn_coord_pred_loss / T
        kl_loss = ARCH.KYPT_KL_LOSS_SCALE * torch.sum(kl_divergence[:, detached_timesteps:], dim=1)  # [B,]


        things = {
            'obs_kypts': observed_keypoints,  # [B, T, K, 3]
            'gaussian_maps': F.interpolate(gaussian_maps.sum(dim=2).clamp(0.0, 1.0), \
                                size=(gaussian_maps.size(-1) * 4, gaussian_maps.size(-1) * 4),
                                mode='bilinear', align_corners=True)[:, :, None].detach(),  # [B, T, 1, 64, 64]
            'pred_kypts': predicted_keypoints,  # [B, T, K, 3]
            'kypt_recon_loss': image_loss,  # [B,]
            'kypt_sep_loss': separation_loss,  # [B,]
            'kypt_coord_pred_loss': vrnn_coord_pred_loss,  # [B,]
            'kypt_reg_loss': heatmap_reg_losses,  # [B,]
            'kypt_kl_loss': kl_loss,  # [B,]
        }
        return things

    # TODO (chmin): add infer keypoints method, for a single step kypt inference.
    def infer_keypoints(self, mix, action, is_first=False, global_step=0):
        """
        From bg_things['agent'], we infer the keypoints on the robot's end-effector.
        The model input: mix
                         action
                         is_first: flag for the first frame (used for keypoint learning)
                         global_step
                  output: reconstructed_images, observed_keypoints, predicted_keypoints
        args:
            agent_seq: A seq of agent layer embedded in the background. -> m_agent * c_agent
        """
        # --- Vision model ---
        # 1) get keypoint from raw observation.
        # In: mixture_recon [1, 3, 64, 64]
        # Out: obs_kypt [1, 3, K]
        # TODO (chmin): this method does not require  
        if is_first: # cache the observation of first step
            self.first_frame = mix # [B, 3, 64, 64]

        # Out - obs_kypts [1, K, 3], heatmap_reg_losses [B, T]
        observed_keypoints, _, _ = self.img_to_kypt(mix) # Out - kedefypoints, heatmaps
        
        if is_first: # cache the keypoint of first step
            self.first_kypt = observed_keypoints # [1, K, 3]

        # 2) reconstruct images from keypoints.
        # Out - [B, T, 3, 64, 64]
        # reconstructed_images, gaussian_maps = self.kypt_to_img(keypoints=observed_keypoints,  # [B, T, K, 3]
        #                                             first_frame=self.first_frame,  # [B, 3, H, W]
        #                                             first_frame_keypoints=self.first_kypt)  # [B, K, 3]
        things = {
            'obs_kypts': observed_keypoints  # [B, T, K, 3]
            # 'gaussian_maps': F.interpolate(gaussian_maps.sum(dim=1)[:, None].clamp(0.0, 1.0), \
            #     size=(gaussian_maps.size(-1) * 4, gaussian_maps.size(-1) * 4),
            #     mode='bilinear', align_corners=True),  # [B, T, 1, 64, 64]
        }
        return things



class ImgToKyptNet(nn.Module):
    """
    Extract K keypoints from given observation.
    """

    def __init__(self):
        nn.Module.__init__(self)

        # build feature extractor
        self.keypoint_feature = self.build_kypt_image_encoder()
        self.features_to_keypoint_heatmaps = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, out_channels=ARCH.NUM_KEYPOINTS,
                kernel_size=1, stride=1, padding=0),
            nn.Softplus()
        )
        self.heatmap_reg_weights = nn.Identity()

    def build_kypt_image_encoder(self,
                                 initial_num_filters=32, output_map_width=16,
                                 layers_per_scale=1):
        """Extracts feature maps from images.
            Returns instance of keypoint feature extractor.
        The encoder iteratively halves the resolution and doubles the number of
        filters until the size of the feature maps is output_map_width by
        output_map_width.
        Args:
            input_shape: Shape of the input image (without batch dimension).
            initial_num_filters: Number of filters to apply at the input resolution.
            output_map_width: width of heatmap -> 16
        Raises:
            ValueError: If the width of the input image is not compatible with
            output_map_width, i.e. if input_width/output_map_width is not a perfect
            square.
        """
        layers = []
        # conv1~3: expand feature dim, conserve size
        # conv1 has 2 additional channel to RGB.
        layers.append(nn.Conv2d(in_channels=3 + 2, out_channels=32, kernel_size=3, stride=1, padding=1))  # H, W = 64
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))  # H, W = 64
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))  # H, W = 64
        layers.append(nn.LeakyReLU())
        # conv4: reduce resolution; conv5~6: keep dims
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))  # H, W = 32
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # H, W = 32
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # H, W = 32
        layers.append(nn.LeakyReLU())
        # conv7: reduce resolution; conv8~9: keep dims
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))  # H, W = 16
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # H, W = 16
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # H, W = 16
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)  # output dim should be 16.

    def forward(self, bg_seq, leverage=False, model_T=0):
        """Builds a model that encodes an image sequence into a keypoint sequence.
        Works as an encoder model.
        The model applies the same convolutional feature extractor to all images in
        the sequence. The feature maps are then reduced to num_keypoints heatmaps, and
        the heatmaps to (x, y, scale)-keypoints.
        Args:
            bg_seq: A sequence of backgroud images of shape [B, T, C, H, W]
            image_shape: Image shape tuple: (num_timesteps, H, W, C).
        Returns:
            A tf.keras.Model object.
        """
        # TODO (chmin): should be compatiable for both single-step and sequential inference.
        # Adjust channel number to account for add_coord_channels:
        infer = True if len(bg_seq.size()) == 4 else False
        # reshape bg_seq to iterate over timesteps
        if infer:
            obs = add_coord_channels(bg_seq)  # [B, 3+2, H, W] -> 2 coord channels are added.
            encoded = self.keypoint_feature(obs)  # [B, feat_channel, 16, 16]
            heatmaps = self.features_to_keypoint_heatmaps(encoded)  # [B, K, 16, 16] K = # kypts
            keypoints = maps_to_keypoints(heatmaps)  # [B, K, 3]
            # Combine timesteps:
            heatmap_reg_loss = None
            detached_timesteps = 0
        else:
            bg_seq = bg_seq.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]
            heatmaps_list = []
            heatmap_reg_losses_list = []
            keypoints_list = []
            # Image to keypoints:

            T = bg_seq.size(0)
            detached_timesteps = T - model_T
            for t, obs in enumerate(bg_seq):  # iterate over sequence
            
                noisy_train_context = torch.no_grad() if leverage and t < detached_timesteps \
                    else nullcontext()

                if t == detached_timesteps:
                    rest_here = 0 # TODO (chmin): remove this after debuging.

                with noisy_train_context:
                    obs = add_coord_channels(obs)  # [B, 3+2, H, W] -> 2 coord channels are added.
                    encoded = self.keypoint_feature(obs)  # [B, feat_channel, 16, 16]
                    heatmaps = self.features_to_keypoint_heatmaps(encoded)  # [B, K, 16, 16] K = # kypts
                    heatmap_reg_loss = self.sparse_reg_heatmap(heatmaps)
                    heatmap_reg_losses_list.append(heatmap_reg_loss)
                    keypoints = maps_to_keypoints(heatmaps)  # [B, K, 3]
                    heatmaps_list.append(heatmaps)
                    keypoints_list.append(keypoints)
            bg_seq = bg_seq.permute(0, 1, 2, 3, 4)  # [B, T, C, H, W]
            # Combine timesteps: heatmaps.sum(1)[:, None]

            heatmaps = torch.stack(heatmaps_list, dim=1)  # [B, T, K, G, G]
            heatmap_reg_loss = torch.stack(heatmap_reg_losses_list, dim=1)  # [B, T]
            keypoints = torch.stack(keypoints_list, dim=1)  # [B, T, K, 3]
        return (keypoints, heatmaps, heatmap_reg_loss)

    def sparse_reg_heatmap(self, heatmaps):
        return get_heatmap_penalty(heatmaps, ARCH.HEATMAP_REG)


class KyptToImgNet(nn.Module):
    """
    Decode images from the extracted keypoints and heatmaps
    """

    def __init__(self):
        nn.Module.__init__(self)
        # image encoder to extract appearance from the first frame
        self.appearance_feature_extractor = self.build_kypt_image_encoder()
        # image decoder goes from Gaussian maps to reconstructed images
        self.image_decoder = self.build_kypt_image_decoder()

        self.adjust_channels_of_decoder_input = nn.Sequential(
            nn.Conv2d(in_channels=128 + 2 * ARCH.NUM_KEYPOINTS + 2, out_channels=128, kernel_size=(1, 1)),
            nn.LeakyReLU()
        )
        self.adjust_channels_of_output_image = nn.Sequential(  # ! no activation
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1)),
            # nn.Sigmoid()
        )

    def build_kypt_image_encoder(self,
                                 initial_num_filters=32, output_map_width=16,
                                 layers_per_scale=1):
        """Extracts feature maps from images.
            Returns instance of keypoint feature extractor.
        The encoder iteratively halves the resolution and doubles the number of
        filters until the size of the feature maps is output_map_width by
        output_map_width.
        Args:
            input_shape: Shape of the input image (without batch dimension).
            initial_num_filters: Number of filters to apply at the input resolution.
            output_map_width: width of heatmap -> 16
        Raises:
            ValueError: If the width of the input image is not compatible with
            output_map_width, i.e. if input_width/output_map_width is not a perfect
            square.
        """
        layers = []
        # conv1~3: expand feature dim, conserve size
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))  # H, W = 64
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))  # H, W = 64
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))  # H, W = 64
        layers.append(nn.LeakyReLU())
        # conv4: reduce resolution; conv5~6: keep dims
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))  # H, W = 32
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # H, W = 32
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # H, W = 32
        layers.append(nn.LeakyReLU())
        # conv7: reduce resolution; conv8~9: keep dims
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))  # H, W = 16
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # H, W = 16
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # H, W = 16
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)  # output dim should be 16.

    def build_kypt_image_decoder(self):
        """Decodes images from feature maps.
        The decoder iteratively doubles the resolution, and halves the number of
        filters until the size of the feature maps the same with original image.
        shape of keypoint sequence: [T, 16, K, 3]
        """
        dec_input_shape = [16, 16, 128]
        layers = []
        # interpolate (16, 16) -> (32, 32), size perserving & feature dim reducing convs
        # layers.append(Interpolate(size=(dec_input_shape[0] * 2, dec_input_shape[1] * 2), mode='bilinear'))
        layers.append(nn.Upsample(size=(dec_input_shape[0] * 2, dec_input_shape[1] * 2), mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # (32, 32, 64)
        layers.append(nn.LeakyReLU())
        layers.append(nn.Upsample(size=(dec_input_shape[0] * 4, dec_input_shape[1] * 4), mode='bilinear', align_corners=False))
        # layers.append(Interpolate(size=(dec_input_shape[0] * 4, dec_input_shape[1] * 4), mode='bilinear'))
        layers.append(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))  # (64, 64, 32)
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, keypoints, first_frame, first_frame_keypoints, leverage=False, model_T=0):
        """
        Reconstructs an image sequence from a keypoint sequence.
        param keypoints: [B, T, K, 3]
        Model architecture:
            (keypoint_sequence, image[0], keypoints[0]) --> reconstructed_image_sequence

        For all frames frames image[t] we also concatenate the Gaussian maps for the
        keypoint obtained from the initial frame image[0]. This helps the decodder
        "impaint" the image regions that are occluded by objects in the first frame.
        Args:

        """
        infer = True if len(keypoints.size()) == 3 else False
        # get the features and maps for first frame:
        # Note that we cannot use the Gaussian maps above because the
        # first-frame_keypoints may be different than the keypoints
        # i.e. (obs vs pred.)

        first_frame_features = self.appearance_feature_extractor(first_frame)  # [1, 128, 16, 16]
        first_frame_gaussian_maps = keypoints_to_maps(first_frame_keypoints)  # [1, K, 16, 16]
        if infer:
            # ! 1) convert keypoints to pixel maps
            guassian_map_sequences = keypoints_to_maps(keypoints)  # In - [1, K, 3] Out - [1, K, 16, 16]
            # ! 2) reconstruct image
            # In: [1, K, 16, 16], [1, 128, 16, 16], [1, K, 16, 16] -> Out: [1, 128 + 2K, 16, 16]
            combined_representation = torch.cat([guassian_map_sequences, first_frame_features,
                first_frame_gaussian_maps], dim=1)  # concat inputs along feature dim
            combined_representation = add_coord_channels(combined_representation) # Out: [1, 128 + 2K + 2, 16, 16]
            combined_representation = self.adjust_channels_of_decoder_input(
            combined_representation)  # Out: [1, 128, 16, 16]
            decoded_representation = self.image_decoder(combined_representation) # Out: [1, 32, 64, 64]
            image_sequences = self.adjust_channels_of_output_image(decoded_representation) # [1, 3, 64, 64]
        else:
            keypoints = keypoints.permute(1, 0, 2, 3)  # [T, B, K, 3]
            image_list = []
            gaussian_maps_list = []
            # iterate over timesteps

            T = keypoints.size(0)
            detached_timesteps = T - model_T

            for t, keypoint in enumerate(keypoints):
                # ! 1) convert keypoints to pixel maps

                noisy_train_context = torch.no_grad() if leverage and t < detached_timesteps \
                    else nullcontext()

                if t ==detached_timesteps:
                    rest_here = 0

                with noisy_train_context: # gaussian_maps_ext.sum(1)[:, None]
                    gaussian_maps = keypoints_to_maps(keypoint)  # In - [B, K, 3] Out - [B, K, 16, 16]
                    gaussian_maps_list.append(gaussian_maps)
                    # ! 2) reconstruct image
                    # In: [B, K, 16, 16], [B, 128, 16, 16], [B, K, 16, 16] -> Out: [B, 128 + 2K, 16, 16]
                    combined_representation = torch.cat([gaussian_maps, first_frame_features,
                                                        first_frame_gaussian_maps], dim=1)  # concat inputs along feature dim
                    combined_representation = add_coord_channels(combined_representation)  # Out: [B, 128 + 2K + 2, 16, 16]
                    combined_representation = self.adjust_channels_of_decoder_input(
                        combined_representation)  # Out: [B, 128, 16, 16]
                    decoded_representation = self.image_decoder(combined_representation)  # Out: [B, 32, 64, 64]
                    image_list.append(self.adjust_channels_of_output_image(decoded_representation))  # Out: [B, 3, 64, 64]
            # combine timesteps
            image_sequences = torch.stack(image_list, dim=1)  # [B, ] -> difference.
            guassian_map_sequences = torch.stack(gaussian_maps_list, dim=1).detach()  # [B
            # add in the first frame of the sequence such that the model only needs to
            # predict the change from the first frame
            # In: [B, T, 3, 64, 64], [B, 1, 3, 64, 64]-> Out: [B, T, 3, 64, 64]
            # image_sequences = torch.add(image_sequences, first_frame[:, None, ...]).clamp(0.0, 1.0)
            image_sequences = torch.add(image_sequences, first_frame[:, None, ...])

        return image_sequences, guassian_map_sequences


class KyptDynaNet(nn.Module):
    """
    Transits keypoints through timestep, via RNN
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.kypt_rnn_cell = nn.GRUCell(input_size=ARCH.NUM_KEYPOINTS * 3 + ARCH.KYPT_Z_DIM,
                                        hidden_size=ARCH.KYPT_RNN_UNITS)

        self.kypt_prior_feat = self.build_kypt_prior_net()
        self.kypt_post_feat = self.build_kypt_post_net()
        # initial keypoint RNN parameters
        self.kypt_rnn_hidden_state = nn.Parameter(torch.rand(1, ARCH.KYPT_RNN_UNITS))
        # TODO (chmin): it should be deprecated.
        self.kypt_rnn_cell_state = nn.Parameter(torch.rand(1, ARCH.KYPT_RNN_UNITS))

        self.kypt_rnn_backup_state = nn.Parameter(torch.rand(1, ARCH.KYPT_RNN_UNITS))

        # Sample a belief from the distribution and decode it into keypoints:
        self.sampler = SampleBestBelief(
            ARCH.NUM_SAMPLES_FOR_BOM,
            use_mean_instead_of_sample=ARCH.USE_DETERMINISTIC_BELIEF)

    def build_kypt_post_net(self):
        """Incorporates observed information into the latent belief.
        rnn_state[t-1], observed_keypoints[t] --> posterior_mean[t], posterior_std[t]
        Args:
            cfg: Hyperparameter ConfigDict.
        Returns:
            Keras Model object.
        """
        layers = []
        layers.append(nn.Linear(in_features=ARCH.KYPT_RNN_UNITS + 3 * ARCH.NUM_KEYPOINTS + ARCH.ACTION_ENHANCE,
                                out_features=ARCH.KYPT_POST_HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=ARCH.KYPT_POST_HIDDEN_DIM,
                                out_features=2 * ARCH.KYPT_Z_DIM))  # rnn_state + keypoints
        return nn.Sequential(*layers)

    def build_kypt_prior_net(self):
        """Computes the prior belief over current keypoints, given past information.
        rnn_state[t-1] --> prior_mean[t], prior_std[t]
        Args:
            cfg: Hyperparameter ConfigDict.
        Returns:
            Keras Model object.
        """
        # rnn_state = tf.keras.Input(shape=[cfg.num_rnn_units], name='rnn_state')
        layers = []
        layers.append(nn.Linear(in_features=ARCH.KYPT_RNN_UNITS + ARCH.ACTION_ENHANCE,
                                out_features=ARCH.KYPT_PRIOR_HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=ARCH.KYPT_PRIOR_HIDDEN_DIM,
                                out_features=2 * ARCH.KYPT_Z_DIM))
        return nn.Sequential(*layers)

    def encode(self, keypoints, state=None, leverage=False, backup=False, model_T=0):
        """Builds the VRNN dynamics model.
        The model takes observed keypoints from the keypoint detector as input and
        returns keypoints decoded from the model's latent belief.
        The model only uses
        the observed keypoints for the first cfg.observed_steps steps. For remaining
        cfg.predicted_steps steps, it predicts keypoints using only the dynamics
        model.
        * conditional prediction of the keypoints.
        Args:
            keypoints: observed keypoints [B, T, K, 3]
            cfg: Hyperparameter ConfigDict.
        Returns:
            A tf.keras.Model and the KL loss tensor.
        """
        B, T, K, _ = keypoints.size()

        rnn_state = self.kypt_rnn_hidden_state.expand(B, ARCH.KYPT_RNN_UNITS)  # [B, Hk]

        # Build model components. All of the weights are shared across observed and
        num_timesteps = T  # observed_steps + predicted_steps
        # keypoint input
        # Initialize loop variables:
        # kypt_rnn_cell_state = self.kypt_rnn_cell_state.expand(B, ARCH.KYPT_RNN_UNITS) # [B, Ck]
        output_keypoints_list = [None] * num_timesteps
        kl_div_list = [None] * T

        detached_timesteps = T - model_T
        for t in range(T):  # iterate over total time steps
            noisy_train_context = torch.no_grad() if leverage and t < detached_timesteps \
                else nullcontext()

            if t == detached_timesteps:
                rest_here = 0 # TODO (chmin): remove this after debuging.

            with noisy_train_context:
                output_keypoints_list[t], rnn_state, kl_div_list[t] = self.vrnn_iteration(
                    input_keypoints=keypoints[:, t], rnn_state=rnn_state,
                    posterior_net=True, state=state[:,t])
        
        # In - len T list of [B, K, 3] -> out - [B, T, K, 3]
        output_keypoints_stack = torch.stack(output_keypoints_list, dim=1)
        kl_div_stack = torch.stack(kl_div_list, dim=1)  # [B, T]

        if backup: # TODO (chmin): check if it works
            self.kypt_rnn_backup_state.data = self.kypt_rnn_hidden_state

        return [output_keypoints_stack, kl_div_stack]

    def generate(self, keypoints, cond_steps, state=None):
        """
        """
        B, T, K, _ = keypoints.size()

        # Build model components. All of the weights are shared across observed and
        # predicted timesteps:
        # rnn_out = self.kypt_rnn_cell(keypoints.view(B*T*K, 3))

        prior_hidden = self.kypt_prior_feat(keypoints)
        # do reparam trick
        self.kypt_post_feat()
        # keypoint decoder

        # ! I won't use scheduled sampler
        # scheduled_sampler_obs = ScheduledSampling(
        #     p_true_start=cfg.scheduled_sampling_p_true_start_obs,
        #     p_true_end=cfg.scheduled_sampling_p_true_end_obs,
        #     ramp_steps=cfg.scheduled_sampling_ramp_steps)
        # scheduled_sampler_pred = ScheduledSampling(
        #     p_true_start=cfg.scheduled_sampling_p_true_start_pred,
        #     p_true_end=cfg.scheduled_sampling_p_true_end_pred,
        #     ramp_steps=cfg.scheduled_sampling_ramp_steps)

        # Format inputs:
        num_timesteps = T  # observed_steps + predicted_steps
        # keypoint input
        kypt_input_shape = [T, K, 3]  # or [T, 3, K]

        # Initialize loop variables:
        kypt_rnn_hidden_state = self.kypt_rnn_hidden_state.expand(B, ARCH.KYPT_RNN_UNITS)
        # kypt_rnn_cell_state = self.kypt_rnn_cell_state.expand(B, ARCH.KYPT_RNN_UNITS)
        output_keypoints_list = [None] * num_timesteps
        kl_div_list = [None] * cond_steps

        for t in range(T):  # iterate over total time steps
            if t < cond_steps:
                output_keypoints_list[t], rnn_state, kl_div_list[t] = self.vrnn_iteration(
                    input_keypoints=keypoints[:, t], rnn_state=kypt_rnn_hidden_state,
                    posterior_net=True, state=state)
            else:
                output_keypoints_list[t], rnn_state, _ = self.vrnn_iteration(
                    input_keypoints=keypoints[:, t], rnn_state=kypt_rnn_hidden_state, state=state)
        # In - len T list of [B, K, 3] -> out - [B, T, K, 3]
        output_keypoints_stack = torch.stack(output_keypoints_list, dim=1)
        kl_div_stack = torch.stack(kl_div_list, dim=1)  # [B, cond_steps]

        return [output_keypoints_stack, kl_div_stack]

    def vrnn_iteration(self,
                       input_keypoints,
                       rnn_state,
                       posterior_net=False, state=None):
        """Performs one timestep of the VRNN.
        Args:
            cfg: ConfigDict with model hyperparameters.
            input_keypoints: [batch_size, K, 3] tensor (one timestep of
            the sequence returned by the keypoint detector).
            rnn_state: Previous recurrent state. [B, Zk]
            dynamics model.
            scheduled_sampler: Keras layer instance that performs scheduled sampling.
            posterior_net: (Optional) A tf.keras.Model that computes the posterior
            latent belief, given observed keypoints and the previous RNN state. If no
            posterior_net is supplied, prior latent belief is used for predictions.
        Returns:
            Three tensors: The output keypoints, the new RNN state, and the KL
            divergence between the prior and posterior (None if no posterior_net is
            provided).
        """
        B, K, _ = input_keypoints.size()  # B, 3, K
        # shape = input_keypoints.shape.as_list()[1:3]
        input_keypoints = input_keypoints.view(B, 3 * K)  # ! for posterior sampling

        # Obtain parameters mean, std for the latent belief distibution:
        prior_out = self.kypt_prior_feat(torch.cat((rnn_state, state), dim=-1))  # In: [B, Hk], Out: [B, 2 * Zk]
        prior_mean, prior_std = torch.split(prior_out, [ARCH.KYPT_Z_DIM] * 2, dim=1)  # each [B, Zk]
        prior_std = F.softplus(prior_std) + 1e-4
        z_kypt_prior = Normal(prior_mean, prior_std)
        if posterior_net:  # T < cond_steps
            # In: [B, H_k + 3 * K], Out: [B, ]
            posterior_out = self.kypt_post_feat(torch.cat([rnn_state, input_keypoints, state], dim=-1))
            post_mean, post_std = torch.split(posterior_out, [ARCH.KYPT_Z_DIM] * 2, dim=1)
            post_std = F.softplus(post_std) + 1e-4
            z_kypt_post = Normal(post_mean, post_std)
            kypt_kl_divergence = kl_divergence(z_kypt_post, z_kypt_prior).sum(dim=1)  # [B,]
        else:  # T >= cond_steps
            # For prior sampling, we don't train the network.
            # predicted steps. Since a reconstruction error is still generated, we
            # need to stop the gradients explicitly to ensure the prior net is not
            post_mean = prior_mean.detach()
            post_std = prior_std.detach()
            kypt_kl_divergence = None
        # Out - [B, Zm], [B, K*3]
        latent_belief, output_keypoints_flat = self.sampler(  # sample by best belief
            [post_mean, post_std, rnn_state, input_keypoints])
        output_keypoints = output_keypoints_flat.view(B, K, 3)  # [B, K, 3]

        # Step the RNN forward:
        keypoints_for_rnn = output_keypoints_flat  # we don't give observed keypoints to RNN
        rnn_input = torch.cat([keypoints_for_rnn, latent_belief], dim=-1)  # [B, Zk + 3*K]

        rnn_state = self.kypt_rnn_cell(rnn_input, rnn_state)  # [B, Hm]
        # rnn_state = rnn_state[0]  # rnn_cell needs state to be wrapped in list.
        return output_keypoints, rnn_state, kypt_kl_divergence


class SampleBestBelief(nn.Module):
    """Chooses the best keypoints from a number of latent belief samples.
    This layer implements the "best of many" sample objective proposed in
    https://arxiv.org/abs/1806.07772.
    "Best" is defined to mean closest in Euclidean distance to the keypoints
    observed by the vision model.
    Attributes:
        num_samples: Number of samples to choose the best from.
        coordinate_decoder: torch.nn object that decodes the latent belief
            into keypoints 'self.kypt_dec' object.
        use_mean_instead_of_sample: If true, do not sample, but just use the mean of
        the latent belief distribution.
    """

    def __init__(self,
                 num_samples,
                 use_mean_instead_of_sample=False):
        nn.Module.__init__(self)
        self.num_samples = num_samples
        self.coordinate_decoder = self.build_belief_to_kypt_dec()
        self.use_mean_instead_of_sample = use_mean_instead_of_sample
        self.uses_learning_phase = True

    def forward(self, x):
        # [B, Zk], [B, Zk], [B, Zk], [B, Hk], [B, 3 * K]
        latent_mean, latent_std, rnn_state, observed_keypoints_flat = x

        # Draw latent samples:
        if self.use_mean_instead_of_sample:  # deterministic sampling
            sampled_latent = torch.repeat(latent_mean, (self.num_samples, 1))  # repeat along the batch axis
        else:
            distribution = Normal(loc=latent_mean, scale=latent_std)  # [B, Zk]
            sampled_latent = distribution.rsample(sample_shape=(self.num_samples,))  # [N_samples, B, Zk]
        sampled_latent_list = torch.unbind(sampled_latent, dim=0)  # len N_samples of [B, Zk]

        # Decode samples into coordinates:
        # sampled_keypoints has shape [num_samples, batch_size, 3 * num_keypoints].
        sampled_keypoints = torch.stack(
            [self.coordinate_decoder(torch.cat([rnn_state, latent], dim=-1))
             for latent in sampled_latent_list],
            dim=0)  # each decoded outputs [B, 3 * K] -> stack -> [N_samples, B, 3 * K]

        # If we have only 1 sample, we can just return that:
        if self.num_samples == 1:
            return [sampled_latent_list[0], sampled_keypoints[0]]
        # Compute L2 prediction loss for all samples (note that this includes both
        # the x,y-coordinates and the keypoint scale):
        sample_losses = torch.mean(
            (sampled_keypoints - observed_keypoints_flat[None, ...]) ** 2.0,
            dim=-1)  # [num_samples, B]

        # Choose the sample based on the loss:
        return self._choose_sample(sampled_latent, sampled_keypoints, sample_losses)

    def build_belief_to_kypt_dec(self):
        """
        Decodes keypoints from the latent belief.
        rnn_state[t-1], latent_z[t] --> keypoints[t]
        """
        layers = []
        layers.append(nn.Linear(ARCH.KYPT_RNN_UNITS + ARCH.KYPT_Z_DIM, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 3 * ARCH.NUM_KEYPOINTS))
        return nn.Sequential(*layers)

    def compute_output_shape(self, input_shape):
        return [input_shape[-1], input_shape[0]]

    def _choose_sample(self, sampled_latent, sampled_keypoints, sample_losses):
        """Returns the first or lowest-loss sample, depending on learning phase.
        During training, the sample with the lowest loss is returned.
        During inference, the first sample is returned without regard to the loss.
        Args:
            sampled_latent: [num_samples, batch_size, latent_code_size] tensor.
            sampled_keypoints: [num_samples, batch_size, 3 * num_keypoints] tensor.
            sample_losses: [num_samples, batch_size] tensor.
        Returns:
            Two tensors: latent and keypoint representation of the best sample.
        """

        # Find the indices of the samples with the lowest loss:
        best_sample_ind = torch.argmin(sample_losses, dim=0).int()  # [B,]
        batch_ind = torch.arange(0, sample_losses.size(1), dtype=torch.int32, device=sampled_latent.device)  # [B,]
        # first rank: the best sample index; second rank: index for batch [0, 1, .., B-1]

        # https://discuss.pytorch.org/t/batched-index-select-tf-gather-nd/27402
        best_latent = sampled_latent[best_sample_ind.long(), batch_ind.long()]  # [B, Zk]
        best_keypoints = sampled_keypoints[best_sample_ind.long(), batch_ind.long()]  # [B, 3 * K]

        # During training, return the best sample. During inference, return the
        # first sample:
        if self.training:  # if traiing mode
            return [best_latent, best_keypoints]
        else:
            return [sampled_latent[0], sampled_keypoints[0]]
