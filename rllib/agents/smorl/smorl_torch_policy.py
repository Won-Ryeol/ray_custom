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


def compute_smorl_loss(seq,
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
    """Constructs loss for the SMORL objective
        Note that the representation learning model for SMORL is 
        SCALOR.
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

    if list(model.value_targ1_high.parameters())[0].requires_grad:
        for ac_params in list(model.value_targ1_high.parameters()) + list(model.value_targ2_high.parameters()) + \
            list(model.value_targ_low.parameters()):
            ac_params.requires_grad_(False)
    # TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin) TODO (chmin)
    #! update target networks via polyak averaging
    if model.global_step > ARCH.JOINT_TRAIN_SMORL_START and model.global_step % ARCH.UPDATE_TARGET_EVERY == 0:
        cur_nets = list(model.value1_high.parameters()) + list(model.value2_high.parameters()) + list(model.value_low.parameters())
        targ_nets = list(model.value_targ1_high.parameters()) + list(model.value_targ2_high.parameters()) + \
        list(model.value_targ_low.parameters())
        for p, p_targ in zip(cur_nets, targ_nets):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

    model.anneal(model.global_step)

    if model.global_step >= ARCH.JOINT_TRAIN_SMORL_START:
        if model.global_step == ARCH.JOINT_TRAIN_SMORL_START:
            print(bcolors.FAIL + "GATSBI Vanilla Pretraining is done @ {0}. \
            Now finetunes GATSBI with exploration.".format(ARCH.JOINT_TRAIN_SMORL_START) + bcolors.ENDC)
        
        if model.global_step == ARCH.JOINT_TRAIN_SMORL_START:
            print(bcolors.FAIL + "GATSBI Vanilla only training is done @ {0}. \
            Now starts joint training with actor-critic learning.".format(ARCH.JOINT_TRAIN_SMORL_START) + bcolors.ENDC)
       
        if model.global_step >= ARCH.JOINT_TRAIN_SMORL_START:
            # if not list(model.actor_high.parameters())[0].requires_grad:
            # TODO (chmin): add smorl-related parameters.

            # if not list(model.reward.parameters())[0].requires_grad:
            #     for reward_params in list(model.reward.parameters()) + list(model.sub_reward.parameters()):
            #         reward_params.requires_grad_(True)

            # if not list(model.bg_module.parameters())[0].requires_grad:
            #     for gswm_params in list(
            #             set(model.parameters()) - set(model.actor_high.parameters()) - \
            #             set(model.value1_high.parameters()) - set(model.value2_high.parameters()) - set(
            #             model.actor_low.parameters()) - set(model.value_low.parameters())
            #             ):
            #         gswm_params.requires_grad_(True)
    else: # pretraining steps
        # TODO (chmin): add smorl-related parameters.
        #* freeze actor and critic networks
        if list(model.actor_high.parameters())[0].requires_grad:
            for ac_params in list(model.actor_high.parameters()) + list(model.value1_high.parameters()) + \
                list(model.value2_high.parameters()) + list(model.actor_low.parameters()) + \
                list(model.value_low.parameters()):
                ac_params.requires_grad_(False)

        #* representation models should be trained.
        if not list(model.bg_module.parameters())[0].requires_grad:
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
            # implement log_summary of gswm.
            log_gif = log_summary(model=model, batch=(eval_obs, eval_action),
                global_step=model.global_step, indices=ARCH.INDICES, device=seq.device,
                cond_steps=ARCH.COND_STEPS, fg_sample=ARCH.FG_SAMPLE, bg_sample=ARCH.BG_SAMPLE, 
                num_gen=ARCH.NUM_GEN)
        print("visualization done for model step {0}".format(model.global_step + 1) + bcolors.ENDC)
    else:
        eval_batch = None

    if model.global_step % ARCH.VIS_EVERY == 0:
        print(bcolors.FAIL + "Current PID is {0}".format(os.getpid()) + bcolors.ENDC)
        print(bcolors.WARNING + "Current model step is {0}".format(model.global_step))
        print("Agent slot is {0}".format(model.agent_slot_idx) + bcolors.ENDC)

    # TODO (chmin): adjust model weights.
    model_weights = list(
            set(model.parameters()) - set(model.actor.parameters())
         - set(model.value.parameters())
        )
    critic_weights = list(model.value_low.parameters())

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.step() # make increment for global_step
    
    if len(seq.size()) == 4:
        #! test dummy loss by rllib. Just make excetion handling.
        seq = seq[:, None, ...].repeat(1, 2, 1, 1, 1) + torch.ones(1).cuda(device=seq.device) # expand temporal dim
        action = action[:, None].repeat(1, 2, 1) # expand temporal dim
        reward = reward[:, None].repeat(1, 2) # expand temporal dim
        # backup k-th step's data to compute new gradient for the student ats (k+1)-th step
    B, T, C, H, W = seq.size()

    # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin)
    # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin)
    # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin) # TODO (chmin)
    # TODO (chmin): below is copy & paste from smorl code.

    # we have our own code convention but here we follow the original code convention of SMORL :)
    bs = seq.size(0) # B
    seq_len = seq.size(1) # T
    device = seq.device
    model.init(seq)
    kl_z_pres_all = seq.new_zeros(bs, seq_len)
    kl_z_what_all = seq.new_zeros(bs, seq_len)
    kl_z_where_all = seq.new_zeros(bs, seq_len)
    kl_z_depth_all = seq.new_zeros(bs, seq_len)
    kl_z_bg_all = seq.new_zeros(bs, seq_len)
    log_imp_all = seq.new_zeros(bs, seq_len)
    log_like_all = seq.new_zeros(bs, seq_len)
    y_seq = seq.new_zeros(bs, seq_len, 3, H, W)
    for i in range(seq_len):
        action = action[:, i]
        x = seq[:, i]
        # img_enc = img_enc_seq[:, i]
        kl_z_bg, kl_z_pres, kl_z_what, kl_z_where, kl_z_depth, log_imp, log_like, y, _  = model.one_step(x, action, eps=eps)
        kl_z_bg_all[:, i] = kl_z_bg 
        kl_z_pres_all[:, i] = kl_z_pres
        kl_z_what_all[:, i] = kl_z_what
        kl_z_where_all[:, i] = kl_z_where
        kl_z_depth_all[:, i] = kl_z_depth
        # TODO (Chmin): replace every argparsing with ARCH.
        if not model.training and ARCH.PHASE_NLL:
            log_imp_all[:, i] = log_imp
        log_like_all[:, i] = log_like
        y_seq[:, i] = y

    counting = torch.stack(model.counting_list, dim=1)

    # TODO (chmin): these are returns from a single 'forward' call. use these to compute
    # TODO total loss instead.
    # y_seq
    log_like_ = log_like_all.flatten(start_dim=1).mean(dim=1)
    kl_z_what_ = kl_z_what_all.flatten(start_dim=1).mean(dim=1)
    kl_z_where_ = kl_z_where_all.flatten(start_dim=1).mean(dim=1)
    kl_z_depth_ = kl_z_depth_all.flatten(start_dim=1).mean(dim=1)
    kl_z_pres_ = kl_z_pres_all.flatten(start_dim=1).mean(dim=1)
    kl_z_bg_ = kl_z_bg_all.flatten(start_dim=1).mean(dim=1)
    log_imp = log_imp_all.flatten(start_dim=1).sum(dim=1)
    counting
    # model.log_disc_list
    # model.log_prop_list
    # model.scalor_log_list
    # loss_dict should contain model, actor, and critic losses.
    return loss_dict

def gaussian_likelihood(obs, fg, bg, alpha_map, global_step):

    raise NotImplementedError
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
