from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
from numpy.core.getlimits import _discovered_machar
import torch
from torch.cuda import is_available

import numpy as np
from typing import Any, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType

#* import reference
torch, nn = try_import_torch()
from torch import distributions as td
from .utils import Linear, TanhBijector, scale_action
from ray.rllib.agents.doctor_mf.utils import Linear, TanhBijector
# GATSBI model related modules.

# import GATSBI vanilla related submodules.
from ray.rllib.agents.doctor_mf.modules.mix import *
from ray.rllib.agents.doctor_mf.modules.module import anneal
from ray.rllib.agents.doctor_mf.modules.arch import ARCH
from ray.rllib.agents.doctor_mf.modules.mix import MixtureModule
from ray.rllib.agents.doctor_mf.modules.obj import ObjModule
from ray.rllib.agents.doctor_mf.modules.keypoint import KeypointModule
from ray.rllib.agents.doctor_mf.modules.utils import bcolors

from .utils import scale_action
# import visualizer of training GATSBI.
from IQA_pytorch import SSIM

ActFunc = Any

# Reward Model (PlaNET), and Value Function
class DenseDecoder(nn.Module):
    """Fully Connected network that outputs a distribution for calculating log_prob
    Used later later in GATSBILoss
    """

    def __init__(self,
                  input_size: int,
                  output_size: int,
                  layers: int,
                  units: int,
                  dist: str = "normal",
                  act: ActFunc = None,
                  sigma=0.1):
        """Initializes FC network
        Args:
            input_size (int): Input size to network
            output_size (int): Output size to network
            layers (int): Number of layers in network
            units (int): Size of the hidden layers
            dist (str): Output distribution, parameterized by FC output logits
            act (Any): Activation function
        """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.act = act
        if not act:
            self.act = nn.ELU
        self.dist = dist
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.layers = []
        cur_size = input_size
        for _ in range(self.layrs):
            self.layers.extend([Linear(cur_size, self.units), self.act()])
            cur_size = units
        self.layers.append(Linear(cur_size, output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        if self.output_size == 1:
            x = torch.squeeze(x)
        if self.dist == "normal":
            output_dist = td.Normal(x, self.sigma)
        elif self.dist == "binary":
            output_dist = td.Bernoulli(logits=x)
        else:
            raise NotImplementedError("Distribution type not implemented!")
        return td.Independent(output_dist, 0)

class RewardDecoder(nn.Module):
    """Fully Connected network that outputs a distribution for calculating log_prob
    Used later later in GATSBILoss
    """

    def __init__(self,
                  input_size: int,
                  output_size: int,
                  layers: int,
                  units: int,
                  dist: str = "normal",
                  act: ActFunc = None,
                  sigma=0.1):
        """Initializes FC network
        Args:
            input_size (int): Input size to network
            output_size (int): Output size to network
            layers (int): Number of layers in network
            units (int): Size of the hidden layers
            dist (str): Output distribution, parameterized by FC output logits
            act (Any): Activation function
        """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.act = act
        if not act:
            self.act = nn.ELU
        self.dist = dist
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.layers = []
        cur_size = input_size
        for _ in range(self.layrs):
            self.layers.extend([Linear(cur_size, self.units), self.act()])
            cur_size = units
        self.layers.append(Linear(cur_size, output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        if self.output_size == 1:
            x = torch.squeeze(x)
        if self.dist == "normal":
            output_dist = td.Normal(x, self.sigma)
        elif self.dist == "binary":
            output_dist = td.Bernoulli(logits=x)
        else:
            raise NotImplementedError("Distribution type not implemented!")
        return td.Independent(output_dist, 0)


class OneHotDist(td.OneHotCategorical):

  def __init__(self, probs=None, logits=None):
    super().__init__(probs=probs, logits=logits)

  def gsample(self, sample_shape=torch.Size()):
    # Straight through biased gradient estimator.
    sample_shape = torch.Size(sample_shape)
    indices = self._categorical.sample(sample_shape)
    num_events = self._categorical._num_events
    probs = self._categorical.probs
    indices = torch.nn.functional.one_hot(indices, num_events).to(probs)
    indices = indices + (probs - probs.detach())
    return indices



# Represents dreamer policy
class ActionDecoder(nn.Module):
    """ActionDecoder is the policy module in GATSBI. It outputs a distribution
    parameterized by mean and std, later to be traed by a custom
    TanhBijector in utils.py for GATSBI.
    """

    def __init__(self,
                  input_size: int,
                  action_size: int,
                  layers: int,
                  units: int,
                  dist: str = "tanh_normal",
                  act: ActFunc = None,
                  min_std: float = 1e-4,
                  init_std: float = 1.0,
                  mean_scale: float = 1.0):
        """Initializes Policy
        Args:
          input_size (int): Input size to network
          action_size (int): Action space size
          layers (int): Number of layers in network
          units (int): Size of the hidden layers
          dist (str): Output distribution, with tanh_normal implemented
          act (Any): Activation function
          min_std (float): Minimum std for output distribution
          init_std (float): Intitial std
          mean_scale (float): Augmenting mean output from FC network
        """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.dist = dist
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.action_size = action_size

        self.layers = []
        self.softplus = nn.Softplus()

        # MLP Construction
        cur_size = input_size
        for _ in range(self.layrs):
            self.layers.extend([Linear(cur_size, self.units), self.act()])
            cur_size = self.units
        if self.dist == "tanh_normal":
            self.layers.append(Linear(cur_size, 2 * action_size))
        elif self.dist == "onehot":
            self.layers.append(Linear(cur_size, action_size))
        self.model = nn.Sequential(*self.layers)

    # Returns distribution
    def forward(self, x):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        # In: x - concat of detr. sto. latent. [B, 230]
        # Out: multiple layers 
        x = self.model(x)
        if self.dist == "tanh_normal":
            # [B, A]
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = torch.tanh(mean)
            std = self.softplus(std + raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            transforms = [TanhBijector()]
            # apply the Tanh bijective transformation.
            dist = td.transformed_distribution.TransformedDistribution(
                dist, transforms)
            # action is indep. along the action dim.
            dist = td.Independent(dist, 1)
        elif self.dist == "onehot":
            dist = OneHotDist(logits=x)
        return dist


    def get_pre_activation(self, x):
        return self.model(x)


class GATSBIVanModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, agent_slot_idx=0):
        """
            Vanilla version of GATSBI.
            TODO (chmin): refactoring required after publication.
        """
        super().__init__(obs_space, action_space, num_outputs, 
                model_config, name)

        nn.Module.__init__(self)
        self.action_size = action_space.shape[0] # 8

        # True if the agent slot if found.
        self.slot_find_flag = False
        self.agent_slot_idx = agent_slot_idx

        self.T = ARCH.T[0]
        self.DETACHED_T = ARCH.DETACHED_T[0]

        self.obj_module = ObjModule()
        self.mixture_module = MixtureModule(action_dim=self.action_size)
        self.keypoint_module = KeypointModule()

        #! vanilla gatsbi has no agent depth.
        # dimension of concatenation of structured latent variables.
        # [1, K * Zm + K * Zc + N * Z(w+w+p...) } + K * Hm + K * Hc + H(w+w+p...)]

        # TODO (chmin): vanilla agent should not have deph!

        self.feat_dim = (ARCH.K * ARCH.Z_MASK_DIM +   # agent mask
                            2 + # agent pos + agent_depth
                            ARCH.MAX *(
                                ARCH.Z_SHIFT_DIM +  # ao-rel-pos
                                ARCH.Z_DEPTH_DIM +  # ao-rel-depth
                                ARCH.Z_WHAT_DIM  # object_shape
                            )
                        )

        self.reward_feat_dim = (ARCH.K * ARCH.Z_MASK_DIM +   # agent mask
                            2 + # agent pos + agent_depth
                            ARCH.MAX *(
                                ARCH.Z_SHIFT_DIM +  # ao-rel-pos
                                ARCH.Z_DEPTH_DIM +  # ao-rel-depth
                                ARCH.Z_WHAT_DIM  # object_shape
                            )
                        )

        print(bcolors.WARNING + "Agent feature dimension is {0}".format(self.feat_dim) + bcolors.ENDC)

        self.reward = RewardDecoder(self.reward_feat_dim, 
                1, 4, ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                sigma=1.0)

        # 'one-hot' policy that outputs the attention for each object.
        self.actor_high = ActionDecoder(self.feat_dim,
                ARCH.MAX, 4, ARCH.DENSE_HIDDEN_DIM,
                dist="onehot")

        # TODO (chmin): check if this type of value learning is a proper choice.
        self.value1_high = DenseDecoder(self.feat_dim, 1, 4,
                        ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                        sigma=ARCH.VALUE_SIGMA)

        self.value2_high = DenseDecoder(self.feat_dim, 1, 4,
                        ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                        sigma=ARCH.VALUE_SIGMA)

        self.value_targ1_high = DenseDecoder(self.feat_dim, 1, 4,
                        ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                        sigma=ARCH.VALUE_SIGMA)

        self.value_targ2_high = DenseDecoder(self.feat_dim, 1, 4,
                        ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                        sigma=ARCH.VALUE_SIGMA)

        self.actor_critic_sub_feat_dim = (ARCH.K * ARCH.Z_MASK_DIM +   # agent mask
                            2 +  # agent pos + agent_depth
                        ARCH.MAX * ARCH.Z_SHIFT_DIM +  # object-pos + rel-pos to others
                            ARCH.Z_DEPTH_DIM +  # ao-rel-depth
                            ARCH.Z_WHAT_DIM +  # object_shape
                            1# occlusion 
                            # ARCH.MAX # sub policy index
                        )

        self.actor_low = ActionDecoder(self.actor_critic_sub_feat_dim + ARCH.MAX,
                ARCH.ACTION_DIM, 4, ARCH.DENSE_HIDDEN_DIM,
                dist="tanh_normal")

        self.value_low = DenseDecoder(self.actor_critic_sub_feat_dim, 
                1, 4, ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                sigma=ARCH.VALUE_SIGMA)

        self.value_targ_low = DenseDecoder(self.actor_critic_sub_feat_dim, 
                1, 4, ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                sigma=ARCH.VALUE_SIGMA)

        # TODO (chmin): let's leverage module list.
        # self.subpolicies = nn.ModuleList([
        #     SubPolicy(n_actions, obs_shape) for _ in range(n_subpolicies)
        # ])

        self.state = None
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))

        # TODO(chmin): create the submodules 

        self.reward_sub_feat_dim = (ARCH.K * ARCH.Z_MASK_DIM +   # agent mask
                            2 + # agent pos + agent_depth
                        ARCH.MAX * ARCH.Z_SHIFT_DIM +  # object-pos + rel-pos to others
                            ARCH.Z_DEPTH_DIM +  # ao-rel-depth
                            ARCH.Z_WHAT_DIM # object_shape
                        )

        self.sub_reward = RewardDecoder(self.reward_sub_feat_dim, 
                1, 4, ARCH.DENSE_HIDDEN_DIM, act=nn.CELU,
                sigma=1.0)

        # stddev of the entire observation.
        self.sigma = ARCH.SIGMA
        # global step to track the optimization process
        # TODO (chmin): check if global_step is set properly
        self.global_step = model_config["global_step"] # it denotes optimization step.
        self.episodic_step = 0 # it increases every policy inference.
        self.start_infer_flag = False # Flag to start policy inference after prefill steps.
        print(bcolors.OKGREEN + "Model is initialized with global step {0}".format(self.global_step) + bcolors.ENDC)

        self.z_agent_mean = 0.0
        self.z_agent_std = 0.0



    def set_infer_flag(self, flag):
        """
        Change the flag for inference start.
        We need this for the keypoint module working right after the prefill step.
        """
        self.start_infer_flag = flag

    def step(self):
        self.global_step += 1

    def set_model_step(self, step):
        self.global_step = step

    def anneal(self, global_step):
        """
        Parse the global training steps to each submodule.
        """
        # self.global_step = global_step
        self.obj_module.anneal(global_step)
        self.mixture_module.anneal(global_step)
        self.keypoint_module.anneal(global_step)

        assert len(ARCH.T) == len(ARCH.T_MILESTONES) + 1, 'len(T) != len(T_MILESTONES) + 1'
        assert len(ARCH.DETACHED_T) == len(ARCH.DETACHED_T_MILESTONES) + 1, 'len(DETACHED_T) != len(DETACHED_T_MILESTONES) + 1'
        
        i = 0
        while i < len(ARCH.T_MILESTONES) and global_step > ARCH.T_MILESTONES[i]:
            i += 1

        j = 0
        while j < len(ARCH.DETACHED_T_MILESTONES) and global_step > ARCH.DETACHED_T_MILESTONES[j]:
            j += 1

        self.T = ARCH.T[i]
        self.DETACHED_T = ARCH.DETACHED_T[j]

    def z_agent_mean_std(self, z_agent, eps=1e-3):
        B, T, D = z_agent.size()
        mean = z_agent.reshape(B*T, D).mean(0,keepdims=True)
        std = z_agent.reshape(B*T, D).std(0,keepdims=True) + eps
        self.z_agent_mean = mean
        self.z_agent_std = std
        return mean, std

    def random_crop(self, seq, T, start=False):
        """
            Sample a subsequence of length T
            Args:
                seq: (B, Told, 3, H, W)
                T: length
            Returns:
                seq: (B, T, 3, H, W)
        """
        if start: # crop the action sequence.
            # action is given as a_{{t-1}:t_{t+T}} for action conditioning
            return seq[:, start:start + T], start

    @torch.no_grad()
    def find_agent_slot(self, kypt, masks, comps=None):
        # kypt: [B, T, 1, H, W]
        # masks: [B, T, K, 1, H, W]
        # with torch.no_grad():
        #     K = masks.size(2)
        #     kypt = torch.stack([kypt] * K, dim=2)
        #     crit = ((kypt > 0.8) * (masks > 0.8)).sum(-1).sum(-1).sum(-1)
        #     tot = (masks > 0.8).sum(-1).sum(-1).sum(-1)
        #     dist = crit / (tot + 1e-5)
        #     _, si = torch.sort(dist, 2, descending=True)
        #     si = si.float()    masks.reshape(-1, 1, 64, 64) 
        #     first = int(si[..., 0].mean().round())  comps.reshape(-1, 3, 64, 64)
        # return first kypt * comps[:, ]
        B, T, K, _, H, W = masks.size()
        comps = comps.reshape(B * T, K, 3, H, W)
        masks = masks.reshape(B * T, K, 1, H, W)
        kypt = kypt.reshape(B * T, 1, H, W)
        criterion = SSIM(channels=1)
        _, indices = torch.sort(torch.tensor([criterion(masks[:, k], kypt, as_loss=False).mean() 
            for k in range(K)],device=masks.device), descending=True)
        return int(indices[0])

    def get_feature_for_agent(self, z_masks, z_comps, z_objs, 
            h_masks, h_comps, h_objs, action=None):
        """
            Constructs feature for input to reward, decoder, actor and critic.
            inputs consist of posterior history and stochastic latents of the scene.        
        """
        
        agent_idx = self.agent_slot_idx
        bg_indices = torch.tensor([k for k in range(ARCH.K) if k != agent_idx], device=z_masks.device).long()

        # process agent-obj depth and position
        obj_position = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
            + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., 2:] # [B, (T), N, 2]
        obj_depth =  z_objs[..., ARCH.Z_PRES_DIM:ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM] # [B, (T), N, 2]
        obj_depth = torch.sigmoid(-obj_depth)  
        # TODO (chmin): don't forget z_occ

        z_agent = z_masks[..., agent_idx, :] # [B, T, Z]
        z_bgs = z_masks[..., bg_indices, :]
        h_agent = h_masks[..., agent_idx, :] # [B, T, H]
        obj_pres = z_objs[..., :1] # [B, T, N, 1]
        obj_scale = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
                + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., :2]  # [B, T, N, 2]
        obj_what = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM:ARCH.Z_PRES_DIM 
                + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM] # [B, T, N, 16]
        # obj pres should be a scale parameter.
        cat_latents = [z_agent, z_bgs, obj_depth, obj_position, obj_what] # [B, T, *]

        # latents_to_cat = [l for (m, l) in zip(cat_latents, latents) if m]     
        if len(z_masks.size()) == 4:
            B, T, K, _ = z_masks.size()
            _, _, N, _ = z_objs.size()
            latents_to_cat = [l.reshape(B, T, -1) for l in cat_latents]
            return torch.cat(latents_to_cat, dim=-1)

        else: #! used in policy inference
            B, K, _ = z_masks.size()
            # squash the entity dimension [B, K * (Hm / Hc / Zm / Zc)]
            # [1, K * Zm + K * Zc + N * Z(w+w+p...) } + K * Hm + K * Hc + H(w+w+p...)]
            latents_to_cat = [l.reshape(B, -1) for l in cat_latents]
            return torch.cat(latents_to_cat, dim=-1)

    def get_indiv_features(self, z_masks, z_comps, z_objs, 
            h_masks, h_comps, h_objs, action=None):
        """
            Constructs feature for input to reward, decoder, actor and critic.
            inputs consist of posterior history and stochastic latents of the scene.        
        """
        
        agent_idx = self.agent_slot_idx
        bg_indices = torch.tensor([k for k in range(ARCH.K) if k != agent_idx], device=z_masks.device).long()

        # process agent-obj depth and position
        obj_position = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
            + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., 2:] # [B, (T), N, 2]
        obj_depth =  z_objs[..., ARCH.Z_PRES_DIM:ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM] # [B, (T), N, 2]
        obj_depth = torch.sigmoid(-obj_depth)  
        # TODO (chmin): don't forget z_occ

        z_agent = z_masks[..., agent_idx, :] # [B, T, Z]
        z_bgs = z_masks[..., bg_indices, :]
        obj_scale = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
                + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., :2]  # [B, T, N, 2]
        obj_what = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM:ARCH.Z_PRES_DIM 
                + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM] # [B, T, N, 16]
        # obj pres should be a scale parameter.

        latent_lists = []
        obj_indices = {ind for ind in range(ARCH.MAX)}

        for idx in range(ARCH.MAX):
            other_objs = list(obj_indices - {idx})

            obj_wise_latents = [z_agent, z_bgs, 
                obj_depth[..., idx, :], obj_position[..., idx, :], obj_what[..., idx, :],
                obj_position[..., idx, :][..., None, :] - obj_position[..., other_objs, :]
            ]
            
            if len(z_masks.size()) == 4:
                B, T, K, _ = z_masks.size()
                _, _, N, _ = z_objs.size()
                latents_to_cat = [l.reshape(B, T, -1) for l in obj_wise_latents]
            else: #! used in policy inference
                B, K, _ = z_masks.size()
                # squash the entity dimension [B, K * (Hm / Hc / Zm / Zc)]
                # [1, K * Zm + K * Zc + N * Z(w+w+p...) } + K * Hm + K * Hc + H(w+w+p...)]
                latents_to_cat = [l.reshape(B, -1) for l in obj_wise_latents]

            obj_wise_latents =  torch.cat(latents_to_cat, dim=-1)
            latent_lists.append(obj_wise_latents)

        return latent_lists

    @torch.no_grad()
    def policy(self, 
            obs: TensorType,
            state: List[TensorType],
            explore=True,
            start_infer_flag=False
            ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """
        Returns the action. Runs through the encoder, recurrent model, and policyMin
        to obtain action.
        The forward inference of the agent.
        Args:
            obs: [B, 3, H, W]
            state: length 15 list of detr and sto states. [
                z_masks, z_comps, h_masks, c_masks, h_comps, c_comps,
                z_pres, z_depth, z_where, z_what, z_dyna, h_objs, c_objs,
                ids, action
            ]
            start_infer_flag: whehter the policy inference has started. i.e. prefill steps -> inference
        """
        # [mean, std, sto, detr., action], of shape [B, D] or [B, Z] or [B, A] 
        if len(state[0].size()) == 4: 
            state = [torch.squeeze(t, 0) for t in state]
            self.episodic_step = 0
            state[-3] = state[-3].long() # exception handling

        if len(obs.size()) == 5:
            obs = obs.squeeze(0)

        if state is None: #! get_initial_state is called outside of this method.
            self.state = self.get_initial_state() 
        else:
            self.state = state
        
        post_mix = state[:6] # z_masks, z_comps, h_masks, c_masks,  h_comps, c_comps
        post_obj = state[6:14] # z_objs, (where, what, ...). h_c_objs, ids
        action = state[-2] # [B, A]
        action = scale_action(action)
        # posterior inference from RNN states.
        is_first_obs = not self.episodic_step or start_infer_flag 

        mixture_out = self.mixture_module.infer(history=post_mix, obs=obs, action=action, 
            episodic_step=self.episodic_step, first=is_first_obs, agent_slot=self.agent_slot_idx)

        obs_diff = obs - mixture_out['bg'] # [1, 3, 64, 64]
        inpt = torch.cat([obs, obs_diff], dim=1) # channel-wise concat
        # TODO (chmin): GATSBI vanilla do not infer the depth of the agent.
        # TODO (chmin): In addition, keypoint inference is not necessary.
        # kypt_out = self.keypoint_module.infer_keypoints(obs, mixture_out['enhanced_act'],
        #     is_first=is_first_obs, global_step=self.global_step)
        obj_out = self.obj_module.infer(history=post_obj, obs=inpt, mix=mixture_out['bg'],
            discovery_dropout=ARCH.DISCOVERY_DROPOUT,
            z_agent=mixture_out['z_masks'][:, self.agent_slot_idx].detach(), # state of the agent.
            h_agent=mixture_out['h_mask_post'][self.agent_slot_idx].detach(), # history of the agent
            enhanced_act=mixture_out['enhanced_act'].detach(),
            first=is_first_obs,
            episodic_step=self.episodic_step
            )
        
        # TODO (chmin): this feature goes for the high-level policy
        feat = self.get_feature_for_agent(
            mixture_out['z_masks'], mixture_out['z_comps'], torch.cat(obj_out['z_objs'], -1),
            mixture_out['h_mask_post'][None], mixture_out['h_comp_post'][None], obj_out['h_c_objs'][0],
            action=action)

        # TODO (chmin): here comes the low-level action inference.

        # obj_out['fg']   mixture_out['bg']
        if self.global_step < ARCH.JOINT_TRAIN_GATSBI_START:
            action = 2.0 * torch.rand(1, self.action_space.shape[0]) - 1.0
            if action[0, 2] > 0 and self.episodic_step < 80:
                action[0, 2] = - action[0, 2] if torch.rand(1) > 0.25 else action[0, 2]
            if action[0, 0] < 0 and self.episodic_step < 80:
                action[0, 0] = - action[0, 0] if torch.rand(1) > 0.25 else action[0, 0]
            logp = torch.zeros((1,), dtype=torch.float32)
        else:
            if ARCH.AGENT_TYPE == 'model_based':
                if self.episodic_step % ARCH.HIGH_LEVEL_HORIZON == 0:
                    action_dist_high = self.actor_high(feat) # [B, N] -> N number of objects
                    # sample by straight-through gradient.
                    sub_policy_idx_inpt = action_dist_high.gsample() # in the form of [1, 0, ...]
                    sub_policy_idx = torch.multinomial(sub_policy_idx_inpt, 1)[0].long() # subpolicy idx in number
                    setattr(self, "sub_policy_idx", sub_policy_idx)
                    setattr(self, "sub_policy_idx_inpt", sub_policy_idx_inpt)

                indiv_feats_list = self.get_indiv_features(
                    mixture_out['z_masks'], mixture_out['z_comps'], torch.cat(obj_out['z_objs'], -1),
                    mixture_out['h_mask_post'][None], mixture_out['h_comp_post'][None], obj_out['h_c_objs'][0],
                    action=action)

                # feature for low-level policy inferred by high-level actor.
                # TODO (chmin): concat with this. 'sub_policy_idx_inpt'
                low_level_feat = torch.cat([indiv_feats_list[self.sub_policy_idx], self.sub_policy_idx_inpt],dim=-1)
                actor_low_dist = self.actor_low(low_level_feat)
                if explore:
                    action = actor_low_dist.sample()
                else:
                    action = actor_low_dist.mean
                logp = actor_low_dist.log_prob(action)
            elif ARCH.AGENT_TYPE == 'model_free': # model-free agent. Do not leverage latent dynamics.
                raise ValueError("Model-free version of GATSBI is not supported yet!")


        # post_mix = state[:4] # z_masks, z_comps, h_masks, h_comps
        # post_obj = state[4:6] # z_objs, (where, what, ...). h_objs, ids
        # action = state[-1]
        self.state = [ mixture_out['z_masks'], mixture_out['z_comps'],
            mixture_out['h_mask_post'][None], mixture_out['c_mask_post'][None], mixture_out['h_comp_post'][None], 
            mixture_out['c_comp_post'][None], * obj_out['z_objs'], * obj_out['h_c_objs'], obj_out['ids']
        ] + [action]

        self.episodic_step += 1 # episodic increment
        return action, logp, self.state

    def imagine_ahead(self, deter_states, sto_states, imagine_horizon, action, raw_action):
        """ Generate imagined frames given the RNN state of masks and comps.
            Args:
                cond_obs: NOTE: we do not need this...
                deter_states:
                sto_states:
                imagine_horizon:
        NOTE that the RNN states should be that of prior.
            actor: the policy network.
            action: action from the last timestep. NOTE that it's enhanced.
        """
        H = imagine_horizon

        h_mask_t_prior, *_ = deter_states

        B, T, K, _ = h_mask_t_prior.size()
        sto_start = []
        for s in sto_states:
            s = s.contiguous().detach() #? should it be contiguous
            shape = [-1] + list(s.size())[2:]
            sto_start.append(s.view(*shape)) # chunk batch and time axes [B, T, K, D]

        deter_start = []
        for d in deter_states:
            d = d.contiguous().detach() #? should it be contiguous
            shape = [-1] + list(d.size())[2:]
            deter_start.append(d.view(*shape)) # chunk batch and time axes [B, T, K, D]
        action = action.detach().reshape(B*T, -1)
        raw_action = raw_action.detach().reshape(B*T, -1)

        # we need to add next_state function, since no action is provided in priori.
        def next_state(states, action, raw_action, cur_horizon):
            """
                First, given (z_t, h_t), infer a_t from policy. 
                Given a_t and (z_t, h_t), update the RNN state to h_{t+1}. 
                Return the updated RNN state
                cur_horizon: current step of the imagination horizon. Necessary for 
                        keypoint inference.
                NOTE: that cur_horizon determines how cur_horizon is related to cond_steps.
            """
            # given h_t and z_t.
            deter_states, sto_states = states # have the shapes of [B*T, *]
            # feature given 
            h_masks_prior, c_masks_prior, h_comps_prior, c_comps_prior, \
                h_objs_prior, c_objs_prior, agent_kypt_mean = deter_states
            z_masks_prior, z_comps_prior, z_objs_prior, ids_prior, \
            proposal, z_occ, z_agent_depth = sto_states
            # split the latents into (z_pres, z_depth, z_where, z_what, z_dyna)
            del deter_states
            del sto_states

            # z_pres: 1, z_depth: 1, z_where: 4 -> fixed & Z_WHAT_DIM and Z_DYNA_DIM are adjustable
            z_pres_prior, z_depth_prior, z_where_prior = z_objs_prior[..., :1], z_objs_prior[..., 1:2], z_objs_prior[..., 2:6]
            z_what_prior, z_dyna_prior = z_objs_prior[..., 6:6 + ARCH.Z_WHAT_DIM], z_objs_prior[..., 6 + ARCH.Z_WHAT_DIM:]

            # get the feature for the policy f_t = [h_t | z_t]
            features, = self.get_feature_for_agent(
                z_masks_prior, z_comps_prior, z_objs_prior,
                h_masks_prior, h_comps_prior, h_objs_prior,
                action=raw_action
            ).detach(), # [B*T, D]

            # let's make sure that the horizon is at least 12 steps.
            if cur_horizon % ARCH.HIGH_LEVEL_HORIZON == 0:
                action_dist_high = self.actor_high(features) # [B, N] -> N number of objects
                sub_policy_ind_inpt = action_dist_high.gsample() # in the form of [1, 0, ...]
                sub_policy_ind = torch.multinomial(sub_policy_ind_inpt, 1) # subpolicy idx in number
                setattr(self, "sub_policy_ind", sub_policy_ind.long().squeeze())
                setattr(self, "sub_policy_ind_inpt", sub_policy_ind_inpt)

            with torch.no_grad():
                indiv_feats_list = self.get_indiv_features(
                    z_masks_prior, z_comps_prior, z_objs_prior,
                    h_masks_prior, h_comps_prior, h_objs_prior,
                    action=raw_action
                ) # [B*T, D]

            low_level_feat_list = []
            for batch_idx in range(features.size(0)): # iterate for B*T dim.
                sub_policy_idx = self.sub_policy_ind[batch_idx]
                low_level_feat = torch.cat([indiv_feats_list[sub_policy_idx][batch_idx], 
                    self.sub_policy_ind_inpt[batch_idx]
                ], dim=-1) # [D]
                low_level_feat_list.append(low_level_feat)
            low_level_feat = torch.stack(low_level_feat_list, dim=0) # [B, D]
            actor_low_dist = self.actor_low(low_level_feat)
            # infer action a_t ~ \pi(a_t | f_t)
            action = actor_low_dist.rsample()
            low_actor_entropy = actor_low_dist.base_dist.base_dist.entropy()
            action = scale_action(action)

            pre_act_action = self.actor_low.get_pre_activation(low_level_feat)
            del features
            del indiv_feats_list

            mixture_out = self.mixture_module.imagine(
                history=(h_masks_prior, c_masks_prior, h_comps_prior, c_comps_prior),
                action=action, # action from the previous step
                z_prevs=(z_masks_prior, z_comps_prior),
                agent_slot=self.agent_slot_idx
                ) # return; z_comps, z_masks, h_mask_prior, h_comp_prior, bg, action

            obj_out = self.obj_module.imagine(
                mix=mixture_out['bg'].detach(), # [B*T, 3, 64, 64], object motion is deterministic.
                history=(h_objs_prior, c_objs_prior), # [B*T, N, D]
                z_prop=((z_pres_prior, z_depth_prior, z_where_prior, z_what_prior, z_dyna_prior),
                         ids_prior, proposal), # [B*T, N,]
                z_agent=mixture_out['z_masks'][:, self.agent_slot_idx].detach(), 
                h_agent=mixture_out['h_mask_prior'][:, self.agent_slot_idx].detach(),
                enhanced_act=mixture_out['enhanced_act'].detach(), sample=True
                )

            deter_states = (mixture_out['h_mask_prior'], mixture_out['c_mask_prior'], mixture_out['h_comp_prior'],
                mixture_out['c_comp_prior'], obj_out['h_c_objs'][0], obj_out['h_c_objs'][1]
            )

            sto_states = (mixture_out['z_masks'], mixture_out['z_comps'], obj_out['z_objs'],
                obj_out['ids'], obj_out['proposal'])

            del mixture_out
            del obj_out

            return deter_states, sto_states, pre_act_action, action, self.sub_policy_ind, \
                self.sub_policy_ind_inpt, low_actor_entropy
        
        # execute imagination rollout.
        sto_last, deter_last = sto_start, deter_start

        # (Will be) len 5 list of lists of len H trajectories; z_masks, z_comps, z_objs, ids, proposal
        sto_outputs = [[] for s in range(len(sto_start))]

        # (Will be) len 6 list of lists of len H trajectories; h_masks, c_masks, h_comps, c_comps, h_objs, c_objs
        deter_outputs = [[] for d in range(len(deter_start))]
        preact_action_list = [] # list of policy output tensors to regularize.
        raw_action_list = [] # list of policy output tensors to regularize.
        sub_policy_idx_list = []
        sub_policy_ind_inpt_list = []
        low_actor_entropy_list = []
        for h in range(H): # imagination horizon h is necessary for the first keypoint
            deter_last, sto_last, preact_action, raw_action, sub_policy_idx, sub_policy_ind_inpt, \
                low_actor_entropy = next_state(states=(deter_last, sto_last), action=action, 
            raw_action=raw_action, cur_horizon=h)
            [so.append(sl) for sl, so in zip(sto_last, sto_outputs)]
            [do.append(dl) for dl, do in zip(deter_last, deter_outputs)]
            preact_action_list.append(preact_action)
            raw_action_list.append(raw_action)
            sub_policy_idx_list.append(sub_policy_idx)
            sub_policy_ind_inpt_list.append(sub_policy_ind_inpt)
            low_actor_entropy_list.append(low_actor_entropy)

        # stack or concat into the shape [H, B*T, ...]
        sto_outputs = [torch.stack(so, dim=0) for so in sto_outputs]
        deter_outputs = [torch.stack(do, dim=0) for do in deter_outputs]
        preact_action = torch.cat(preact_action_list, dim=0) # [B * T * H, A'], A' is action dim before tanh squashing
        raw_action = torch.cat(raw_action_list, dim=0) # [B * T * H, A'], A' is action dim before tanh squashing
        sub_policy_idx = torch.stack(sub_policy_idx_list, dim=0) #  [B * T * H, 1]
        sub_policy_ind_inpt = torch.stack(sub_policy_ind_inpt_list, dim=0) #  [B * T * H, ARCH.MAX]
        low_actor_entropy = torch.stack(low_actor_entropy_list, dim=0) #  [B * T * H, ARCH.MAX]

        # imag feat has both gradient for high- and low- level policies.
        imag_feat = self.get_feature_for_agent(
            sto_outputs[0], sto_outputs[1], sto_outputs[2],
            deter_outputs[0], deter_outputs[2], deter_outputs[4],
            aux=aux_info, action=raw_action)

        indiv_latent_lists = self.get_indiv_features(
            sto_outputs[0], sto_outputs[1], sto_outputs[2],
            deter_outputs[0], deter_outputs[2], deter_outputs[4],
            aux=aux_info, action=raw_action
        )
        
        return imag_feat, preact_action, raw_action, indiv_latent_lists, \
            sub_policy_idx, sub_policy_ind_inpt, low_actor_entropy

    def get_initial_state(self) -> List[TensorType]:
        z_masks = self.mixture_module.z_mask_0.expand(1, ARCH.K, ARCH.Z_MASK_DIM) # 0
        z_comps = self.mixture_module.z_comp_0.expand(1, ARCH.K, ARCH.Z_COMP_DIM) # 1
       
        z_pres = torch.zeros(1, ARCH.MAX, ARCH.Z_PRES_DIM, device=z_masks.device) # 6
        z_depth = torch.zeros(1, ARCH.MAX, ARCH.Z_DEPTH_DIM, device=z_masks.device) # 7
        z_where = torch.zeros(1, ARCH.MAX, ARCH.Z_WHERE_DIM, device=z_masks.device) # 8
        z_what = torch.zeros(1, ARCH.MAX, ARCH.Z_WHAT_DIM, device=z_masks.device) # 9
        z_dyna = torch.zeros(1, ARCH.MAX, ARCH.Z_DYNA_DIM, device=z_masks.device) # 10
        ids = torch.zeros(1, ARCH.MAX, device=z_masks.device).long() # 13, ids_prop
        action = torch.zeros(1, ARCH.ACTION_DIM, device=z_masks.device)

        mix_states = self.mixture_module.get_init_recur_state()
        obj_states = self.obj_module.get_init_recur_state()

        post_states = [z_masks, z_comps]
        prior_states = []

        post_states.extend([t for t in mix_states['post']]) # 2, 3, 4, 5
        post_states.extend([ids, action]) # 13

        return post_states
