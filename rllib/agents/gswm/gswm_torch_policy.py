import enum
import itertools
import logging

from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.catalog import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.agents.gswm.gswm_model import RewardDecoder
from ray.rllib.agents.gswm.utils import FreezeParameters
from ray.rllib.agents.gswm.modules.arch import ARCH
from ray.rllib.agents.gswm.modules.visualize import *
from ray.rllib.agents.gswm.modules.utils import bcolors
from ray.rllib.agents.gswm.modules.utils import spatial_transform
from ray.rllib.agents.gswm.modules.module import anneal
from ray.rllib.agents.gswm.modules.median_pool import MedianPool2d
from ray.rllib.agents.gswm.modules.arcmargin import ArcMarginProduct

import torchvision
import os
FIG_DIR = os.path.expanduser("~/rss22figs/")

# TODO (chmin): arc margin loss for metric learning
from torch.distributions.kl import kl_divergence
import random
import os
import numpy as np
import numbers
from IQA_pytorch import SSIM

from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.policy.policy import Policy

# for customized lr scheduling.
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from contextlib import nullcontext
from .utils import generate_heatmaps

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td

logger = logging.getLogger(__name__)

from torch.distributions import Normal


## policy entropy scheduling.

class ModelLearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule."""

    @DeveloperAPI
    def __init__(self, lr, lr_schedule):
        self.cur_lr = lr
        if lr_schedule is None:
            self.lr_schedule = ConstantSchedule(lr, framework=None)
        else:
            self.lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None)

        self.init_optimizers = self._optimizers

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        self.cur_lr = self.lr_schedule.value(global_vars["timestep"])

        if global_vars["timestep"] <= ARCH.JOINT_TRAIN_GSWM_START and len(self._optimizers) == 3:
            self._optimizers = [self._optimizers[0]] # TODO (chmin): avoid actor-critic learning
        if global_vars["timestep"] == ARCH.JOINT_TRAIN_GSWM_START + 1  and len(self._optimizers) == 1:
            self._optimizers.extend(self.init_optimizers[1:])

        for idx, opt in enumerate(self._optimizers): #* only for model lr
            for p in opt.param_groups: # idx: 0 (model), 1 (actor), 2 (critic)
                if idx == 0: # lr decay for model
                    p["lr"] = self.cur_lr
                    break
                # TODO (chmin): below is deprecated. will be removed in the future.
        if global_vars['timestep'] % 500 == 0:
            print(bcolors.OKBLUE + "Current global_step of model: {0} and \
lr: {1}".format(global_vars['timestep'], self.cur_lr) + bcolors.ENDC)
        # TODO (chmin): schedule entropy here.
        # ENTROPY_DECAY_STEPS
        # This is the computation graph for workers (inner adaptation steps)
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LinearLR

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

def select(kypt_intensity, kypts):
    """

    Args:
        z_pres: (B, N, 1)
        *kargs: each (B, N, *) -> state_post, state_prior, z, ids, proposal

    Returns:ㅡ
        each (B, N, *)
    """
    # Take index
    # 1. sort by z_pres
    # 2. truncate
    # (B, N)
    indices = torch.argsort(kypt_intensity, dim=-1, descending=True) # [..., 0] # sort along G*G dim
    # Truncate
    indices = indices[:, :, :ARCH.MAX]

    # Now use this thing to index every other thing
    def gather(x, indices):
        if len(x.size()) > len(indices.size()):
            indices = indices[..., None].expand(*indices.size()[:3], x.size(-1))
        return torch.gather(x, dim=2, index=indices)

    kypts = transform_tensors(kypts, func=lambda x: gather(x, indices))
    # return the sorted latents w.r.t. z_{pres}
    return kypts

def random_crop(seq, T):
    """
    Sample a subsequence of length T
    Args:
        seq: (B, Told, 3, H, W)
        T: length
    Returns:
        seq: (B, T, 3, H, W)
    """
    obs, action, reward, next_action = seq

    t_seq = obs.size(1)
    assert t_seq >= T, f't_seq: {t_seq}, T: {T}'
    start = random.randint(0, t_seq - T)
    return (obs[:, start:start + T], action[:, start:start + T], 
        reward[:, start:start + T], next_action[:, start:start + T])


def energy( state, action, next_state, no_trans=False, sigma=0.5):
    """Energy function based on normalized squared L2 norm."""

    norm = 0.5 / (sigma**2)

    if no_trans:
        diff = state - next_state
    else:
        pred_trans = self.transition_model(state, action)
        diff = state + pred_trans - next_state

    return norm * diff.pow(2).sum(2).mean(1)

def contrastive_loss(obs, action, next_obs):
    """
        Contrastive loss adopted from C-SWM.
        https://github.com/tkipf/c-swm
    """
    objs = self.obj_extractor(obs)
    next_objs = self.obj_extractor(next_obs)

    state = self.obj_encoder(objs)
    next_state = self.obj_encoder(next_objs)

    # Sample negative state across episodes at random
    batch_size = state.size(0)
    perm = np.random.permutation(batch_size)
    neg_state = state[perm]

    self.pos_loss = self.energy(state, action, next_state)
    zeros = torch.zeros_like(self.pos_loss)
    
    self.pos_loss = self.pos_loss.mean()
    self.neg_loss = torch.max(
        zeros, self.hinge - self.energy(
            state, action, neg_state, no_trans=True)).mean()

    loss = self.pos_loss + self.neg_loss


def compute_gswm_loss(obs,
                        action,
                        reward,
                        next_action,
                        eval_batch,
                        model,
                        imagine_horizon,
                        discount=0.99,
                        lambda_=0.95,
                        cur_lr=3e-4,
                        polyak=0.995,
                        log=False):
    """Constructs loss for the GSWM objective

        Args:
            obs (TensorType): Observations (o_t)
            action (TensorType): Actions (a_(t-1))
            reward (TensorType): Rewards (r_(t-1))
            model (TorchModelV2): GSWMModel, encompassing all other models
            imagine_horizon (int): Imagine horizon for actor and critic loss
            discount (float): Discount
            lambda_ (float): Lambda, like in GAE
            log (bool): If log, generate gifs
        """
    # first few thousand steps are dedicated for keypoint learning
    # crop out the data

    # visualize the imagination obs.reshape(-1, 3, 64, 64)
    # obs - [B, T, 3, 64, 64], action - [B, T, A] (enhanced)

    if list(model.value_targ1_high.parameters())[0].requires_grad:
        for ac_params in list(model.value_targ1_high.parameters()) + list(model.value_targ2_high.parameters()) + \
            list(model.value_targ_low.parameters()):
            ac_params.requires_grad_(False)
    # TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin)
    #! update target networks via polyak averaging
    if model.global_step > ARCH.JOINT_TRAIN_GSWM_START and model.global_step % ARCH.UPDATE_TARGET_EVERY == 0:
        cur_nets = list(model.value1_high.parameters()) + list(model.value2_high.parameters()) + list(model.value_low.parameters())
        targ_nets = list(model.value_targ1_high.parameters()) + list(model.value_targ2_high.parameters()) + \
        list(model.value_targ_low.parameters())
        for p, p_targ in zip(cur_nets, targ_nets):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

    model.anneal(model.global_step)

    if model.global_step >= ARCH.JOINT_TRAIN_GSWM_START:
        if model.global_step == ARCH.JOINT_TRAIN_GSWM_START:
            print(bcolors.FAIL + "GATSBI Vanilla Pretraining is done @ {0}. \
            Now finetunes GATSBI with exploration.".format(ARCH.JOINT_TRAIN_GSWM_START) + bcolors.ENDC)
        
        if model.global_step == ARCH.JOINT_TRAIN_GSWM_START:
            print(bcolors.FAIL + "GATSBI Vanilla only training is done @ {0}. \
            Now starts joint training with actor-critic learning.".format(ARCH.JOINT_TRAIN_GSWM_START) + bcolors.ENDC)
       
        if model.global_step >= ARCH.JOINT_TRAIN_GSWM_START:
            if not list(model.actor_high.parameters())[0].requires_grad:
                for ac_params in list(model.actor_high.parameters()) + list(model.value1_high.parameters()) + \
                 list(model.value2_high.parameters()) + list(model.actor_low.parameters()) + \
                 list(model.value_low.parameters()):
                    ac_params.requires_grad_(True)

            if not list(model.reward.parameters())[0].requires_grad:
                for reward_params in list(model.reward.parameters()) + list(model.sub_reward.parameters()):
                    reward_params.requires_grad_(True)

            if not list(model.mixture_module.parameters())[0].requires_grad:
                for gswm_params in list(
                        set(model.parameters()) - set(model.actor_high.parameters()) - \
                        set(model.value1_high.parameters()) - set(model.value2_high.parameters()) - set(
                        model.actor_low.parameters()) - set(model.value_low.parameters())
                        ):
                    gswm_params.requires_grad_(True)
    else: # pretraining steps
        #* freeze actor and critic networks
        if list(model.actor_high.parameters())[0].requires_grad:
            for ac_params in list(model.actor_high.parameters()) + list(model.value1_high.parameters()) + \
                list(model.value2_high.parameters()) + list(model.actor_low.parameters()) + \
                list(model.value_low.parameters()):
                ac_params.requires_grad_(False)

        #* representation models should be trained.
        if not list(model.mixture_module.parameters())[0].requires_grad:
            for gswm_params in list(
                    set(model.parameters()) - set(model.actor_high.parameters()) - \
                    set(model.value1_high.parameters()) - set(model.value2_high.parameters()) - set(
                    model.actor_low.parameters()) - set(model.value_low.parameters())
                    ):
                gswm_params.requires_grad_(True)

    # empty GPU cache periodically
    if model.global_step % 50 == 0:
        torch.cuda.empty_cache()
         
    log_gif = None
    if log:
        eval_obs, eval_action, _ = eval_batch
        print(bcolors.OKBLUE + "Making visualization ...")
        # visualize the whole episode
        with torch.no_grad():
            log_gif = log_summary(model=model, batch=(eval_obs, eval_action),
                global_step=model.global_step, indices=ARCH.INDICES, device=obs.device,
                cond_steps=ARCH.COND_STEPS, fg_sample=ARCH.FG_SAMPLE, bg_sample=ARCH.BG_SAMPLE, 
                num_gen=ARCH.NUM_GEN)
        print("visualization done for model step {0}".format(model.global_step + 1) + bcolors.ENDC)
    else:
        eval_batch = None

    if model.global_step % ARCH.VIS_EVERY == 0:
        print(bcolors.FAIL + "Current PID is {0}".format(os.getpid()) + bcolors.ENDC)
        print(bcolors.WARNING + "Current model step is {0}".format(model.global_step))
        print("Agent slot is {0}".format(model.agent_slot_idx) + bcolors.ENDC)

    model_weights = list(
        set(model.parameters()) - set(model.actor_high.parameters()) - \
        set(model.value1_high.parameters()) - set(model.value2_high.parameters()) - set(
        model.actor_low.parameters()) - set(model.value_low.parameters()) - \
        set(model.reward.parameters()) - set(model.sub_reward.parameters())
        )

    reward_weights = list(model.reward.parameters()) + list(model.sub_reward.parameters())
    model_weights = model_weights + reward_weights
    critic_weights = list(model.value1_high.parameters()) + list(model.value2_high.parameters()) + \
        list(model.value_low.parameters())
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.step() # make increment for global_step
    # https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/9
    # obs.reshape(-1, 3, 64, 64)
    if ARCH.TRAIN_LONG_TERM and model.global_step >= ARCH.TRAIN_LONG_TERM_FROM and model.global_step < ARCH.JOINT_TRAIN_GATSBI_START:
        # vanilla gatsbi do not leverage noisy rnn states.
        leverage = False
        obs, action, reward, next_action = random_crop((obs, action, reward, next_action), model.T)
        detached_timesteps = 0
    else: # joint training or initial training (not noisy RNN)
        if model.global_step >= ARCH.JOINT_TRAIN_GATSBI_START:
            leverage = False
            obs, action, reward, next_action = random_crop((obs, action, reward, next_action), ARCH.IMAGINE_TIMESTEP_CUTOFF)
            detached_timesteps = 0
        else:
            obs, action, reward, next_action = random_crop((obs, action, reward, next_action), model.T)
            leverage = False
            detached_timesteps = 0
    
    if len(obs.size()) == 4:
        #! test dummy loss by rllib. Just make excetion handling.
        obs = obs[:, None, ...].repeat(1, 2, 1, 1, 1) + torch.ones(1).cuda(device=obs.device) # expand temporal dim
        action = action[:, None].repeat(1, 2, 1) # expand temporal dim
        reward = reward[:, None].repeat(1, 2) # expand temporal dim
        # backup k-th step's data to compute new gradient for the student ats (k+1)-th step
    B, T, C, H, W = obs.size()

    raw_action = action.clone()[:, detached_timesteps:]

    #* mixture module.
    mixture_out = model.mixture_module.encode(
        seq=obs, action=action, agent_idx=model.agent_slot_idx,
        leverage=leverage, model_T=model.T)       
    # used for generating pseudo mask and bg regularization
    obs_diff = obs - mixture_out['bg']
    inpt = torch.cat([obs, obs_diff], dim=2)

    #* do action enhance or not.
    A = action.size(-1)
    if model.global_step < ARCH.KYPT_MIX_JOINT_UNTIL: 
        action = mixture_out['enhanced_act']
    else:
        action = mixture_out['enhanced_act'].detach()
    # prepare for the policy loss.
    #* keypoint module
    if model.global_step < ARCH.JOINT_TRAIN_GATSBI_START:
        kypt_out = model.keypoint_module.predict_keypoints(obs, action, model.global_step,
            leverage=leverage, model_T=model.T)
    else:
        with torch.no_grad():
            kypt_out = model.keypoint_module.predict_keypoints(obs, action, model.global_step,
                leverage=leverage, model_T=model.T)

    if not model.slot_find_flag and model.global_step >= ARCH.MODULE_TRAINING_SCHEME[1]:
        model.slot_find_flag = True
        # find the agent slot from the masks using the keypoint map 
        model.agent_slot_idx = model.find_agent_slot(kypt_out['gaussian_maps'].detach(), 
            mixture_out['masks'].detach(),
            mixture_out['comps'].detach()
            )
        print(bcolors.WARNING + "AGENT SLOT IS FOUND AS {0}".format(model.agent_slot_idx))
        print('============================='+ bcolors.ENDC)

    #* object module
    obj_out = model.obj_module.track(obs = inpt, mix = mixture_out['bg'],
        discovery_dropout = ARCH.DISCOVERY_DROPOUT,
        z_agent = mixture_out['z_masks'][:, :, model.agent_slot_idx].detach(), # state of the agent.
        h_agent = mixture_out['h_masks_post'][:, :, model.agent_slot_idx].detach(), # history of the agent
        enhanced_act = mixture_out['enhanced_act'].detach(),
        agent_mask = mixture_out['masks'][:, :,model.agent_slot_idx].detach(),
        leverage = leverage, model_T=model.T
    )

    B, T, N, D = obj_out['z_where'].size()
    _, _, K, _ = kypt_out['obs_kypts'].size()

    #* fg and bg recons and kl divs for computing elbo.
    fg = obj_out['fg'] # [B, T, 3, H, W] 
    bg = mixture_out['bg'] # [B, T, 3, H, W] 
    # [B, T] -> [B, ]
    kl_fg = obj_out['kl_fg'][:, detached_timesteps:].sum(-1)
    kl_bg = ARCH.BG_KL_SCALE * mixture_out['kl_bg'][:, detached_timesteps:].sum(-1)

    alpha_map = obj_out['alpha_map'] # [B, T, 1, H, W]
    if model.global_step < ARCH.FIX_ALPHA_STEPS:
        center_point = torch.cat([torch.zeros(B * T, 1, 2), torch.ones(B * T, 1, 1)], 
            dim=-1).to(obs.device) # [B * T, 1, 3]
        alpha_map = generate_heatmaps(center_point, sigma=ARCH.ALPHA_MAP_SIGMA, heatmap_width=64)
        alpha_map = (ARCH.FIX_ALPHA_VALUES * alpha_map.reshape(B, T, 1, 64, 64) - \
            kypt_out['gaussian_maps'].detach()).clamp(min=0.0)

    loglikelihood = gaussian_likelihood(obs[:, detached_timesteps:], fg[:, detached_timesteps:],
        bg[:, detached_timesteps:], alpha_map[:, detached_timesteps:], model.global_step)

    #* compute kl_div for both obj  and mixture modules 
    kl = kl_bg if (ARCH.BG_ON and model.global_step < ARCH.MODULE_TRAINING_SCHEME[1]) else kl_bg + kl_fg
    kl = kl # * (ARCH.T[0] / model.T) # maintain the loss scale. 
    elbo = loglikelihood - kl # [B, ]
    
    # visualization
    assert elbo.size() == (B,)
    #* kypt loss
    kypt_loss = torch.zeros(1).to(obs.device)
    if model.global_step < ARCH.JOINT_TRAIN_GATSBI_START:
        kypt_loss = kypt_out['kypt_recon_loss'] + kypt_out['kypt_sep_loss'] + kypt_out['kypt_coord_pred_loss'] \
                + kypt_out['kypt_kl_loss'] + kypt_out['kypt_reg_loss']
    B, T, _, H, W = obs.size()

    #* agent embed loss
    agent_embed_loss = torch.zeros(1).to(obs.device)
    if model.global_step < ARCH.JOINT_TRAIN_GATSBI_START:
        agent_embed_loss = ARCH.EMBED_SCALE * nn.BCELoss(reduction=ARCH.BCE)(
            mixture_out['masks'][:, detached_timesteps:, model.agent_slot_idx] * mixture_out['comps'][:, detached_timesteps:, model.agent_slot_idx].detach(), 
                (kypt_out['gaussian_maps'][:, detached_timesteps:].detach() > 0.1).float() * mixture_out['comps'][:, detached_timesteps:, model.agent_slot_idx].detach())
        agent_embed_loss = agent_embed_loss.sum([2, 3, 4]).mean(1)

    #* --- training scheme ---
    res_reg_loss = torch.norm(mixture_out['mask_residuals'][:, detached_timesteps:], p=2, dim=-1).mean(1).sum(1) + \
            torch.norm(mixture_out['comp_residuals'][:, detached_timesteps:], p=2, dim=-1).mean(1).sum(1)

    #* object discovery encoding regularization loss. the scale of prop_map and x_enc should be similar 
    #* shape of discovery map: [B, D, G, G] 
    obj_disc_scale = torch.norm(obj_out['disc_prop_map'][:, detached_timesteps:], p=2, dim=2).sum([1, 2, 3]).mean(0) / ARCH.T[0]

    if not hasattr(model, "obj_disc_scale_rm"):
        setattr(model, "obj_disc_scale_rm", obj_disc_scale.detach())
    elif model.global_step % 200 == 0:
        model.obj_disc_scale_rm = 0.99 * model.obj_disc_scale_rm + 0.01 * obj_disc_scale.detach()
    obj_disc_reg_loss = torch.norm(obj_disc_scale - model.obj_disc_scale_rm)
    # [B, T, 1, 2], [B, T, 1], [B, T, N]

    #! 1) keypoint learning starts first
    reward_loss = torch.zeros(1).to(obs.device)
    model_loss = kypt_loss
    if model.global_step >= ARCH.KYPT_MIX_JOINT_UNTIL:
        model_loss = torch.zeros_like(kypt_loss, device=obs.device)

    #! 2) keypoint + mixture learning
    if model.global_step >= ARCH.MODULE_TRAINING_SCHEME[0]:
        model_loss = model_loss - elbo

    #! 3) (keypoint) + mixture + object
    if model.global_step >= ARCH.MODULE_TRAINING_SCHEME[1]: # after agent idx is found.
        model_loss = model_loss + ARCH.RESIDUAL_SCALE * res_reg_loss
        model_loss = model_loss + obj_disc_reg_loss
        if model.global_step < ARCH.JOINT_TRAIN_GATSBI_START:
            model_loss = model_loss + agent_embed_loss

    #! 4) reward learning.1
    sub_reward_loss = torch.zeros(1)
    sub_reward_reg = torch.zeros(1).to(obs.device)
    sub_reward_reg_loss = torch.zeros(1).to(obs.device)
    if model.global_step >= ARCH.PRETRAIN_GATSBI_UNTIL and not (reward.mean() == 0):

        features = model.get_feature_for_agent(
            mixture_out['z_masks'][:, detached_timesteps:], mixture_out['z_comps'][:, detached_timesteps:], 
            obj_out['z_objs'][:, detached_timesteps:], mixture_out['h_masks_post'][:, detached_timesteps:], 
            mixture_out['h_comps_post'][:, detached_timesteps:], obj_out['h_objs'][:, detached_timesteps:],
            action=raw_action
        )

        indiv_latent_lists = model.get_indiv_features(
            mixture_out['z_masks'][:, detached_timesteps:], mixture_out['z_comps'][:, detached_timesteps:], 
            obj_out['z_objs'][:, detached_timesteps:], mixture_out['h_masks_post'][:, detached_timesteps:], 
            mixture_out['h_comps_post'][:, detached_timesteps:], obj_out['h_objs'][:, detached_timesteps:],
            action=raw_action
        )

        reward_pred = model.reward(features) # [B, T, ]
        reward_loss = - reward_pred.log_prob(reward[:, detached_timesteps:]).sum(1).mean(0)
        # compute object-wise reward values
        sub_reward_pred = torch.zeros(1).to(obs.device)
        for idx in range(ARCH.MAX):
            sub_reward_pred_dist = model.sub_reward(indiv_latent_lists[idx])
            if model.global_step <= ARCH.SUB_REWARD_REG_STEPS + ARCH.JOINT_TRAIN_GATSBI_START:
                sub_reward_reg = sub_reward_reg + (reward[:, detached_timesteps:][..., None].detach() / ARCH.MAX - \
                    sub_reward_pred_dist.mean[..., None]).norm(dim=-1).sum(1).mean(0)
            sub_reward_pred = sub_reward_pred + sub_reward_pred_dist.mean # [B, T]

        sub_reward_reg_loss = sub_reward_reg
        sub_reward_loss = (reward[:, detached_timesteps:][..., None].detach() - sub_reward_pred[..., None]).norm(dim=-1).sum(1).mean(0)

        model_loss = model_loss + reward_loss + sub_reward_loss + sub_reward_reg_loss
        
    #* finalize model loss
    model_loss = model_loss.mean()

    return_dict = {
        "model_loss": model_loss,
        "reward_loss": reward_loss, # TODO (chmin): this should be critical.
        "model/elbo": elbo.mean(),
        "model/loglikelihood": loglikelihood.mean(),
        "model/kl": kl.mean(),
        "model/sub_reward_loss": sub_reward_loss.mean(),
        "model/sub_reward_reg_loss": sub_reward_reg_loss.mean(),
        "model/obj/kl_fg": kl_fg.mean(),
        "model/obj/kl_pres": obj_out['kl_pres'].mean(),
        "model/obj/kl_depth": obj_out['kl_depth'].mean(),
        "model/obj/kl_where": obj_out['kl_where'].mean(),
        "model/obj/kl_what": obj_out['kl_what'].mean(),
        "model/obj/kl_dyna": obj_out['kl_dyna'].mean(),
        "model/obj/obj_disc_reg_loss": obj_disc_reg_loss.mean(),
        "model/obj/obj_disc_scale": model.obj_disc_scale_rm.mean(),
        "model/kypt/recon": kypt_out['kypt_recon_loss'].mean(),
        "model/kypt/sep": kypt_out['kypt_sep_loss'].mean(),
        "model/kypt/coord_pres": kypt_out['kypt_coord_pred_loss'].mean(),
        "model/kypt/kl": kypt_out['kypt_kl_loss'].mean(),
        "model/kypt/reg": kypt_out['kypt_reg_loss'].mean(),
        "model/alpha_mean": alpha_map.mean(),
        "model/mix/agent_embed_loss": agent_embed_loss.mean(),
        "model/mix/mask_res_norm": torch.norm(mixture_out['mask_residuals'], p=2, dim=-1).mean(1).sum(1).mean(),
        "model/mix/comp_res_norm": torch.norm(mixture_out['comp_residuals'], p=2, dim=-1).mean(1).sum(1).mean(),
        "model/mix/kl_bg": kl_bg.mean(),
        }
    preact_action = torch.zeros(1).to(obs.device)
    actor_loss = torch.zeros(1).to(obs.device)
    critic_loss = torch.zeros(1).to(obs.device)

    if model.global_step >= ARCH.JOINT_TRAIN_GATSBI_START:
        imagine_timestep_cutoff = ARCH.IMAGINE_TIMESTEP_CUTOFF + detached_timesteps
        setattr(model.mixture_module, 'agent_depth_raw_prev', 
            mixture_out['agent_depth_raw'][:, detached_timesteps:imagine_timestep_cutoff].clone().reshape(B*ARCH.IMAGINE_TIMESTEP_CUTOFF, 1)
        )
        # during the pretrain step, skip the actor-critic learning (reduce the memory consumption)
        deter_states = (mixture_out['h_masks_post'][:, detached_timesteps:imagine_timestep_cutoff].clone(), 
                mixture_out['c_masks_post'][:, detached_timesteps:imagine_timestep_cutoff].clone(), 
                mixture_out['h_comps_post'][:, detached_timesteps:imagine_timestep_cutoff].clone(),
                mixture_out['c_comps_post'][:, detached_timesteps:imagine_timestep_cutoff].clone(), 
                obj_out['h_objs'][:, detached_timesteps:imagine_timestep_cutoff].clone(), 
                obj_out['c_objs'][:, detached_timesteps:imagine_timestep_cutoff].clone(),
                agent_kypt_mean[:, :ARCH.IMAGINE_TIMESTEP_CUTOFF].clone()
            )
        sto_states = (mixture_out['z_masks'][:, detached_timesteps:imagine_timestep_cutoff].clone(),
                mixture_out['z_comps'][:, detached_timesteps:imagine_timestep_cutoff].clone(), 
                obj_out['z_objs'][:, detached_timesteps:imagine_timestep_cutoff].clone(),
                obj_out['ids'][:, detached_timesteps:imagine_timestep_cutoff].clone(),
                obj_out['proposal'][:, detached_timesteps:imagine_timestep_cutoff].clone(),
                z_occ_mask[:, :ARCH.IMAGINE_TIMESTEP_CUTOFF].clone(),
                z_agent_depth[:, :ARCH.IMAGINE_TIMESTEP_CUTOFF].clone().detach(),
            )
        with torch.no_grad():
            deter_states = [d.detach() for d in deter_states]
            sto_states = [s.detach() for s in sto_states]
            action = action.detach()
        with FreezeParameters(model_weights):
            imag_feat, preact_action, _, _, \
                _, indiv_latent_lists, sub_policy_ind, sub_policy_ind_inpt, \
                low_actor_entropy = model.imagine_ahead(deter_states=deter_states,  # [H, B*T, D]
                sto_states=sto_states, imagine_horizon=imagine_horizon, 
                action=action[:, detached_timesteps:imagine_timestep_cutoff]
                , raw_action=raw_action[:, :ARCH.IMAGINE_TIMESTEP_CUTOFF])
        with FreezeParameters(model_weights + critic_weights):
            reward = model.reward(imag_feat).mean # [H, B*T]  has gradient.
            reward_splits = torch.split(reward, ARCH.HIGH_LEVEL_HORIZON, 0)
            # high-level has shorter horizon horizon
            # X: high level horizon
            high_level_reward = torch.stack([r.sum(dim=0) for r 
                in reward_splits], dim=0)  # [H // X, B*T]

            # compute low level values
            sub_reward_list = []
            low_level_val_list = []
            low_level_val_targ_list = []

            for temp_idx in range(ARCH.IMAGINE_HORIZON):
                _sub_reward_list = []
                _low_level_val_list = []
                _low_level_val_list_critic = [] # val feat to compute critic loss
                _low_level_val_targ_list = []

                for batch_idx in range(reward.size(1)):
                    sub_policy_idx = sub_policy_ind[temp_idx][batch_idx]
                    low_feat = indiv_latent_lists[sub_policy_idx][temp_idx][batch_idx]

                    # reward -> for actor loss computation. (lambda return)
                    sub_reward = model.sub_reward(low_feat).mean # has grad
                    _sub_reward_list.append(sub_reward)

                    # value -> for actor loss computation. (lambda return)
                    value_low = model.value_low(low_feat).mean # has grad
                    _low_level_val_list.append(value_low)

                    # value_targ
                    with torch.no_grad():
                        value_low_targ = model.value_targ_low(low_feat).mean
                        _low_level_val_targ_list.append(value_low_targ)

                _sub_reward = torch.stack(_sub_reward_list, dim=0)
                _low_level_val = torch.stack(_low_level_val_list, dim=0)
                _low_level_val_targ = torch.stack(_low_level_val_targ_list, dim=0)

                sub_reward_list.append(_sub_reward)
                low_level_val_list.append(_low_level_val)
                low_level_val_targ_list.append(_low_level_val_targ)

            sub_rewards = torch.stack(sub_reward_list, dim=0) # [H, B*T]
            value_low = torch.stack(low_level_val_list, dim=0) # [H, B*T]
            target_value_low = torch.stack(low_level_val_targ_list, dim=0).detach() # [H, B*T]

            # X: high level horizon
            high_level_inds = torch.arange(0, ARCH.IMAGINE_HORIZON, ARCH.HIGH_LEVEL_HORIZON).long()
            imag_feat_high = imag_feat[high_level_inds] # [H // X, B*T]

            # make current high-level value prediction
            value1_high = model.value1_high(imag_feat_high).mean # [H // X, B*T] grad required
            value2_high = model.value2_high(imag_feat_high).mean # [H // X, B*T] grad required
            # take minimum of the two current values. (TD3 style)
            value_high = torch.min(value1_high, value2_high) # [H // X, B*T]
            # compute low level values
        pcont_high = discount * torch.ones_like(high_level_reward) # [H // X , B*T]
        pcont_low = discount * torch.ones_like(sub_rewards) # [H // X, B*T]
        # compute return based on Eq.(5) and Eq.(6) of the paper.
        with torch.no_grad():
            # predict target value with slow value. do not require gradient.
            target_value1_high = model.value_targ1_high(imag_feat_high).mean # [H // X, B*T]
            target_value2_high = model.value_targ2_high(imag_feat_high).mean # [H // X, B*T]

            # take minimum of the two target values. (TD3 style)
            target_value_high = torch.min(target_value1_high, target_value2_high)

            # crop out the first element of sequence.
            target_returns_high = lambda_return(reward=high_level_reward[:-1], value=target_value_high[:-1], 
                pcont=pcont_high[:-1], bootstrap=target_value_high[-1], lambda_=lambda_) # [H-1 // X, B*T]

            target_returns_low = lambda_return(reward=sub_rewards[:-1], value=target_value_low[:-1],
                pcont=pcont_low[:-1], bootstrap=target_value_low[-1], lambda_=lambda_) # [H-1, B*T]

        # compute return w.r.t. current value networks.
        # compute low-level returns to compute critic loss. has gradient
        returns_high = lambda_return(reward=high_level_reward[:-1], value=value_high[:-1], 
            pcont=pcont_high[:-1], bootstrap=value_high[-1], lambda_=lambda_) # [H-1 // X, B*T]

        returns_low = lambda_return(reward=sub_rewards[:-1], value=value_low[:-1],
                pcont=pcont_low[:-1], bootstrap=value_low[-1], lambda_=lambda_) # [H-1, B*T]

        discount_shape = pcont_high[:1].size() # [1, B * T]
        # In- [1, B * T] and pcont[:-2] - shape [H - 2 , B * T] & Out - [H - 1, B * T]
        discount_high = torch.cumprod(
            torch.cat([torch.ones(*discount_shape).to(device), pcont_high[:-2]], dim=0),
                dim=0) # [H - 1, B * T]

        discount_low = torch.cumprod(
            torch.cat([torch.ones(*discount_shape).to(device), pcont_low[:-2]], dim=0),
                dim=0) # [H - 1, B * T]

        # * 1) train high-level actor critics

        if ARCH.HIGH_LEVEL_ACTOR_GRAD == 'dynamics':
            actor_high_loss = - (discount_high * returns_high).mean()
            action_dist = model.actor_high(imag_feat_high[:-1]) # [H // X - , B*T]

        elif ARCH.HIGH_LEVEL_ACTOR_GRAD == 'reinforce':
            action_high_splits = torch.split(sub_policy_ind_inpt, ARCH.HIGH_LEVEL_HORIZON, 0)
            # high-level has shorter horizon horizon
            # X: high level horizon
            action_high = torch.stack([a.sum(dim=0) for a 
                in action_high_splits], dim=0)  # [H // X, B*T]
            action_dist = model.actor_high(imag_feat_high[1:-1]) # [H // X - 2, B*T]
            log_prob_high = action_dist.log_prob(action_high[1:-1]) # [H // X  - 2, B*T]
            # first sequence is useless as no gradient for actor is defined.
            # Actions:     -   [a2]   a3
            # Targets:      -  [t2]
            # Baselines:  [v1]   v2    v3
            # Loss:        -      l2\

            # baseline should be estimated from current step. i.e. t->
            baseline = target_value_high[:-2] # [H // X - 2, B * T]
            advantage = (target_returns_high[1:] - baseline).detach()
            actor_high_loss = - (log_prob_high * advantage).mean() # [H // X - 2, B * T]

        else:
            raise ValueError("Wrong high-level actor arguments!!")

        # [H -1, B * T] * [H -1, B * T] 
        # TODO (chmin): jointly train high & low-level agents.
        actor_low_loss = - (discount_low * returns_low).mean()
        actor_loss = actor_high_loss + actor_low_loss

        actor_loss_log = actor_loss.detach().clone().mean()
        preact_norm_loss = ARCH.PRE_ACT_NORM_WEIGHT * torch.norm(preact_action, p=2, dim=-1).mean()
        actor_loss = actor_loss + preact_norm_loss  # []

        END_POINTS = [(ARCH.JOINT_TRAIN_GATSBI_START, ARCH.EXP_ENTROPY), 
            (ARCH.JOINT_TRAIN_GATSBI_START + ARCH.ENTROPY_DECAY_STEPS, 0.0)]

        def linear_interpolation(left, right, alpha):
            return left + alpha * (right - left)

        def entropy_schedule(cur_step):
            for (l_t, l), (r_t, r) in zip(END_POINTS[:-1], END_POINTS[1:]):
            # When found, return an interpolation (default: linear).
                if l_t <= cur_step < r_t:
                    alpha = float(cur_step - l_t) / (r_t - l_t)
                    return linear_interpolation(l, r, alpha)
            return 0.0 * ARCH.EXP_ENTROPY

        policy_entropy = action_dist.entropy()
        ent_scale = entropy_schedule(model.global_step)
        actor_entropy_loss = - ent_scale * policy_entropy.mean() - ent_scale * low_actor_entropy.mean() # maximize
        actor_loss = actor_loss + actor_entropy_loss

        # Critic Loss
        with torch.no_grad():
            val_high_feat = imag_feat_high.detach()[:-1] # [H // X - 1, B * T]

            target_high = target_returns_high.detach() # [H // X - 1, B * T]
            target_low = target_returns_low.detach() # [H // X - 1, B * T]

            # Get current low-level value features
            low_level_feat_list = []
            for temp_idx in range(ARCH.IMAGINE_HORIZON - 1):
                _low_level_feat_list = []
                for batch_idx in range(reward.size(1)):
                    sub_policy_idx = sub_policy_ind[temp_idx][batch_idx]
                    
                    low_feat = indiv_latent_lists[sub_policy_idx][temp_idx][batch_idx].clone().detach()
                    _low_level_feat_list.append(low_feat)

                _low_level_feat = torch.stack(_low_level_feat_list, dim=0)
                low_level_feat_list.append(_low_level_feat)
            feat_low = torch.stack(low_level_feat_list, dim=0) # [H - 1, B*T]
            val_high_discount = discount_high.detach() # [H // X - 1, B * T]
            val_low_discount = discount_low.detach() # [H - 1, B * T]

        # train current high-level value function, has gradient
        val_high_pred1 = model.value1_high(val_high_feat) # td.Independent; [H // X - 1, B * T]
        val_high_pred2 = model.value2_high(val_high_feat) # td.Independent; [H // X - 1, B * T]

        # train current low-level value function
        val_low_pred = model.value_low(feat_low) #  td.Independent; [H - 1, B * T]

        # negative log likelihood is the same as MSE loss. requires grad.
        critic_high_loss1 = - (val_high_discount * val_high_pred1.log_prob(target_high)).mean() #  sum(0).mean(0)
        critic_high_loss2 = - (val_high_discount * val_high_pred2.log_prob(target_high)).mean() #  sum(0).mean(0)
        critic_high_loss = critic_high_loss1 + critic_high_loss2

        critic_low_loss = - (val_low_discount * val_low_pred.log_prob(target_low)).mean()
        critic_loss = critic_high_loss + critic_low_loss

        if torch.rand(1) > 0.95:
            print(f"Actor loss is {actor_loss_log.mean()} and Critic loss is {critic_loss.mean()}")
        ac_dict = { # do reinforcement learning
            "preact_norm_loss": preact_norm_loss,
            "actor_entropy_loss": actor_entropy_loss,
            "actor_loss": actor_loss,
            "actor_high_loss": actor_high_loss,
            "actor_low_loss": actor_low_loss,
            "actor_loss_log": actor_loss_log,
            "critic_loss": critic_loss,
            "critic_high_loss": critic_high_loss,
            "critic_low_loss": critic_low_loss,
            "critic_loss_log": critic_loss,
        }
    else:
        ac_dict = { # do reinforcement learning
            "preact_norm_loss": torch.zeros(1).to(obs.device).mean(),
            "actor_entropy_loss": torch.zeros(1).to(obs.device).mean(),
            "actor_loss": actor_loss,
            "actor_high_loss": torch.zeros(1).to(obs.device).mean(),
            "actor_low_loss": torch.zeros(1).to(obs.device).mean(),
            "actor_loss_log": torch.zeros(1).to(obs.device).mean(),
            "critic_high_loss": torch.zeros(1).to(obs.device).mean(),
            "critic_low_loss": torch.zeros(1).to(obs.device).mean(),
            "critic_loss": critic_loss,
            "critic_loss_log": critic_loss,


        }

    if not reward.mean() == 0 and torch.rand(1) > 0.95:
        print(f"Reward loss is {reward_loss.mean()} and requires_grad is \
            {list(model.reward.parameters())[0].requires_grad}")
    return_dict.update(ac_dict)

    if log_gif is not None: 
        return_dict["log_gif"] = log_gif

    # TODO (chmin): check the benefit of freeing memories.
    del mixture_out
    del obj_out
    del kypt_out

    return return_dict

def gaussian_likelihood(obs, fg, bg, alpha_map, global_step):

    if ARCH.BG_ON and global_step < ARCH.MODULE_TRAINING_SCHEME[2]:
        recon = bg # bg_only steps
    else:
        # TODO (chmin): alpha_map affects bg...
        recon = fg + (1. - alpha_map) * bg
    # recon: [b, T, 3, 64, 64]
    dist = Normal(recon, ARCH.SIGMA)
    # [B, T, 3, H, W]
    loglikelihood = dist.log_prob(obs)
    # [B, ]
    # loglikelihood = loglikelihood.flatten(start_dim=1).sum(-1)
    loglikelihood = loglikelihood.sum([2, 3, 4]).mean(1) # mean over time seq.

    return loglikelihood

# Similar to GAE-Lambda, calculate value targets
def lambda_return(reward, value, pcont, bootstrap, lambda_):
    """
    Setting lambda=1 gives a discounted Monte Carlo return.
    Setting lambda=0 gives a fixed 1-step return.
        args:
            imagination horizon H.
            reward: a tensor of rewards. [H - 1, B * T]
            value: [H - 1, B * T]
            pcont: tensor of discount factors. [H - 1, B * T] 
            bootstrap: v_H 
            lambda_:
    """
    def agg_fn(x, y):
        """
            args:
                x: the last bootstrapping value.
        """
        return y[0] + y[1] * lambda_ * x
    # next values
    # values from v_{t+1} value[-1]
    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)

    inputs = reward + pcont * next_values * (1 - lambda_)

    last = bootstrap # why bootstrap is the last one?
    returns = []
    for i in reversed(range(len(inputs))): # H-1, H-2, ... ->, 0.
        last = agg_fn(last, [inputs[i], pcont[i]])
        returns.append(last)

    returns = list(reversed(returns))
    returns = torch.stack(returns, dim=0)
    return returns

def gswm_loss(policy, model, dist_class, train_batch):
    log_gif = False
    if "log_gif" in train_batch:
        log_gif = True
        eval_batch = (train_batch["eval_obs"], train_batch["eval_actions"], train_batch["eval_rewards"])
    else:
        eval_batch = None

    policy.stats_dict = compute_gswm_loss(
        train_batch["obs"], # [B, T, 3, 64, 64]
        train_batch["actions"], # [B, T, A]
        train_batch["rewards"], # [B, T, ]
        train_batch["next_actions"], # [B, T, ]
        eval_batch,
        policy.model, # GATSBI vanilla model.
        policy.config["imagine_horizon"], # 15
        policy.config["discount"], # 0.99
        policy.config["lambda"], # 0.95
        policy.cur_lr, # 0.95
        polyak=ARCH.POLYAK,
        log=log_gif
    )

    loss_dict = policy.stats_dict

    return (loss_dict["model_loss"], loss_dict["actor_loss"], loss_dict["critic_loss"])

def build_gswm_model(policy, obs_space, action_space, config):

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        config["gswm_model"],
        name="GSWMModel",
        framework="torch")

    policy.model_variables = policy.model.variables()

    return policy.model


def action_sampler_fn(policy, model, input_dict, state, explore, timestep):
    """Action sampler function has two phases. During the prefill phase,
    actions are sampled uniformly [-1, 1]. During training phase, actions
    are evaluated through GATSBIPolicy and an additive gaussian is added
    to incentivize exploration.
    """
    obs = input_dict["obs"]
    # Custom Exploration
    # set timesteps for our model     
    if timestep <= policy.config["prefill_timesteps"] or ARCH.REAL_WORLD is not 'false':
        logp = torch.zeros((1,), dtype=torch.float32)
        # Random action in space [-1.0, 1.0]
        action = 2.0 * torch.rand(1, model.action_space.shape[0]) - 1.0
        state = model.get_initial_state() # this method is not problematic
        model.set_infer_flag(False)
    else:
        if not model.start_infer_flag:
            model.set_infer_flag(True)
            state = model.get_initial_state()
            action, logp, state = model.policy(obs, state, explore, 
                start_infer_flag=True)
        else:
            action, logp, state = model.policy(obs, state, explore,
                start_infer_flag=False)
        # we use entropy-based exploration instead.
        # action = td.Normal(action, policy.config["explore_noise"]).sample()
        # action = torch.clamp(action, min=-1.0, max=1.0)

    return action, logp, state


def gswm_stats(policy, train_batch):
    return policy.stats_dict


def gswm_optimizer_fn(policy, config):
    model = policy.model

    mix_weights = list(model.mixture_module.parameters())
    obj_weights = list(model.obj_module.parameters())
    kypt_weights = list(model.keypoint_module.parameters()) 
    agent_depth_weights = list(model.agent_depth.parameters()) 
    occl_metric_weights = list(model.occl_metric.parameters())
    reward_weights = list(model.reward.parameters()) + list(model.sub_reward.parameters())
    actor_weights = list(model.actor_high.parameters()) + list(model.actor_low.parameters())
    critic_weights = list(model.value1_high.parameters()) + list(model.value2_high.parameters())
    critic_weights = critic_weights + list(model.value_low.parameters())
    # model, actor, and critic have different learning rates.
    model_opt = torch.optim.Adam(
        agent_depth_weights + occl_metric_weights + 
        mix_weights + obj_weights + kypt_weights + reward_weights,
        lr=config["td_model_lr"])
    actor_opt = torch.optim.Adam(actor_weights, lr=config["actor_lr"])
    critic_opt = torch.optim.Adam(critic_weights, lr=config["critic_lr"])

    return (model_opt, actor_opt, critic_opt)

def before_loss_init(policy: Policy, obs_space, action_space,
                       config: TrainerConfigDict) -> None:
    ModelLearningRateSchedule.__init__(policy, config["td_model_lr"], config["lr_schedule"])

def apply_grad_clipping(policy, optimizer, loss):
    info = {}
    if policy.config["grad_clip"]:
        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                if param_group['lr'] == ARCH.AC_LR:
                    grad_gnorm = nn.utils.clip_grad_norm_(
                        params, ARCH.AC_GRAD_CLIP)
                else:
                    grad_gnorm = nn.utils.clip_grad_norm_(
                        params, policy.config["grad_clip"])
                if isinstance(grad_gnorm, torch.Tensor):
                    grad_gnorm = grad_gnorm.cpu().numpy()
                info["grad_gnorm"] = grad_gnorm
    return info

# helper function to create the policy instance.
GSWMTorchPolicy = build_torch_policy(
    name="GSWMTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.gswm.DEFAULT_CONFIG,
    action_sampler_fn=action_sampler_fn,
    loss_fn=gswm_loss,
    stats_fn=gswm_stats,
    make_model=build_gswm_model,
    optimizer_fn=gswm_optimizer_fn,
    extra_grad_process_fn=apply_grad_clipping,
    # before_init=setup_early_mixins,
    before_loss_init=before_loss_init, #* https://github.com/ray-project/ray/issues/15554
    mixins=[
    ModelLearningRateSchedule,
    ])
