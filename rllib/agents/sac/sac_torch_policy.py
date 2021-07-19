"""
PyTorch policy class used for SAC.
"""

import gym
from gym.spaces import Discrete
import logging
from typing import Dict, List, Optional, Tuple, Type, Union

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.sac.sac_tf_policy import build_sac_model, \
    postprocess_trajectory, validate_spaces
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import \
    TorchDistributionWrapper, TorchDirichlet
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical, TorchSquashedGaussian, TorchDiagGaussian, TorchBeta)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_ops import huber_loss
from ray.rllib.utils.typing import LocalOptimizer, TensorType, \
    TrainerConfigDict


# explainable ai.
from gatsbi_rl.baselines.slide_to_target_config import CFG
from gatsbi_rl.gradcam.saliency_map import *
from gatsbi_rl.rllib_agent.utils import FreezeParameters
from gatsbi_rl.gradcam.lrp import convert_vision
from gatsbi_rl.gradcam.lrp.visualize import *
from gatsbi_rl.gradcam.lrp.converter import SlimFlatten

# visualize as gif.
from gatsbi_rl.gatsbi.visualize import * # import all function

import cv2
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from torch.distributions import Categorical

torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)



def build_sac_model_and_action_dist(
        policy: Policy,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> \
        Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.

    Args:
        policy (Policy): The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config (TrainerConfigDict): The SAC trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    model = build_sac_model(policy, obs_space, action_space, config)
    action_dist_class = _get_dist_class(config, action_space)
    return model, action_dist_class


def _get_dist_class(config: TrainerConfigDict, action_space: gym.spaces.Space
                    ) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.

    Args:
        config (TrainerConfigDict): The Trainer's config dict.
        action_space (gym.spaces.Space): The action space used.

    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        if config["normalize_actions"]:
            return TorchSquashedGaussian if \
                not config["_use_beta_distribution"] else TorchBeta
        else:
            return TorchDiagGaussian


def action_distribution_fn(
        policy: Policy,
        model: ModelV2,
        obs_batch: TensorType,
        *,
        state_batches: Optional[List[TensorType]] = None,
        seq_lens: Optional[TensorType] = None,
        prev_action_batch: Optional[TensorType] = None,
        prev_reward_batch=None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        is_training: Optional[bool] = None) -> \
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    """The action distribution function to be used the algorithm.

    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.

    Args:
        policy (Policy): The Policy being queried for actions and calling this
            function.
        model (TorchModelV2): The SAC specific Model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_policy_output` method.
        obs_batch (TensorType): The observations to be used as inputs to the
            model.
        state_batches (Optional[List[TensorType]]): The list of internal state
            tensor batches.
        seq_lens (Optional[TensorType]): The tensor of sequence lengths used
            in RNNs.
        prev_action_batch (Optional[TensorType]): Optional batch of prev
            actions used by the model.
        prev_reward_batch (Optional[TensorType]): Optional batch of prev
            rewards used by the model.
        explore (Optional[bool]): Whether to activate exploration or not. If
            None, use value of `config.explore`.
        timestep (Optional[int]): An optional timestep.
        is_training (Optional[bool]): An optional is-training flag.

    Returns:
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
            The dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    if "vis" in obs_batch:
        vis_step = obs_batch['vis'].permute(2, 0, 1)
        obs_batch = obs_batch['obs']
    if len(obs_batch.size()) != 4 and CFG.OBS_TYPE == 'vision': # exception for full state
        #* weird handling, but okay. Signal for the start of each episode.
        obs_batch = obs_batch.squeeze(0) # if episode reset; [1, 3, 64, 64]

        if hasattr(model, 'episodic_step'):
            setattr(model, 'vis_episode', model.episode_obs[:, :model.episodic_step]) # slice upto episode length.
            # setattr(model, 'is_vis', True)
        setattr(model, 'episodic_step', 0)
    # TODO(chmin): reset handling for fully observable agent
    if len(obs_batch.size()) == 3 and CFG.OBS_TYPE == 'state':
        obs_batch = obs_batch.squeeze(0) # if episode reset; [1, 3, 64, 64]
        if hasattr(model, 'episodic_step'):
            setattr(model, 'vis_episode', model.episode_obs[:, :model.episodic_step])
            # TODO (chmin): add visualization2
        setattr(model, 'episodic_step', 0)



    # Get base-model output (w/o the SAC specific parts of the network).
    model_out, _ = model({
        "obs": obs_batch,
        "is_training": is_training,
    }, [], None)
    # Use the base output to get the policy outputs from the SAC model's
    # policy components.
    distribution_inputs = model.get_policy_output(model_out)
    # Get a distribution class to be used with the just calculated dist-inputs.
    action_dist_class = _get_dist_class(policy.config, policy.action_space)
    
    # model.episode_obs = 

    # TODO (chmin): optimize conditioning.
    if hasattr(model, 'episodic_step') and CFG.OBS_TYPE == 'vision':
        if CFG.FRAME_STACK:
            obs_batch = obs_batch[:, -3:]
        model.episode_obs[:, model.episodic_step] = obs_batch # [1, 3, 64, 64]
        model.episodic_step += 1        
    if hasattr(model, 'episodic_step') and CFG.OBS_TYPE == 'state':
        model.episode_obs[:, model.episodic_step] = vis_step # [1, 3, 64, 64]
        model.episodic_step += 1        




    return distribution_inputs, action_dist_class, []


def actor_critic_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """
    Constructs the loss for the Soft Actor Critic.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Should be True only for debugging purposes (e.g. test cases)!
    
    action_size = train_batch['actions'].size()[-1]

    # for idx, ik_error in enumerate(train_batch['infos']):
    #     if ik_error:
    #         for dim in range(action_size):
    #             train_batch['actions'][idx][dim] = 0.0

    deterministic = policy.config["_deterministic_loss"]
    obs_raw = train_batch[SampleBatch.CUR_OBS]
    model_out_t, _ = model({
        "obs": obs_raw,
        "is_training": True,
    }, [], None)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }, [], None)

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
        policy_t = torch.exp(log_pis_t)
        log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        q_t = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(),
            num_classes=q_t.size()[-1])
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = \
            (1.0 - train_batch[SampleBatch.DONES].float()) * \
            q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else \
            action_dist_t.deterministic_sample()
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                                 policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * \
            q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS] +
        (policy.config["gamma"]**policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [
        torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))
    ]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error)))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).

    # mlp_lastlayer_weight = model.action_model.action_out._model._modules['0'].weight.data

    action_dist_norm = torch.norm(action_dist_t.dist.loc, p=2)
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach())
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t.detach()),
                dim=-1)) + CFG.ACT_REG_WEIGHT * action_dist_norm
    else:
        alpha_loss = -torch.mean(model.log_alpha *
                                 (log_pis_t + model.target_entropy).detach())
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy) + CFG.ACT_REG_WEIGHT * action_dist_norm
    
        # #* compute auxiliary loss
        # # raw_out = model.get_policy_output(model_out_t)

        # raw_action = action_dist_t.deterministic_sample() # [B, A]
        # raw_cart_x = CFG.ACTION_SCALE * raw_action[:, 0]
        # raw_cart_y = CFG.ACTION_SCALE * raw_action[:, 1]
        # raw_cart_z = CFG.Z_SCALE * raw_action[:, 2]

        # boundary = torch.tensor([-0.0004, 0.0004], device=obs_raw.device)
        # x_idx = torch.bucketize(raw_cart_x, boundary) # [B, ]
        # y_idx = torch.bucketize(raw_cart_y, boundary) # [B, ]
        # z_idx = torch.bucketize(raw_cart_z, boundary) # [B, ]

        # idx = (3**2 * x_idx + 3 * y_idx + z_idx).long()

        # act_logit = model.act_disc_model(model_out_t)
        # # act_aux_loss = Categorical(logits=act_logit) # [B, 27]
        # act_disc_loss = nn.CrossEntropyLoss()(act_logit, idx.long())
        actor_loss = actor_loss  # + act_disc_loss

        # _, act_pred = torch.max(act_logit, dim=1)

        # x_idx_pred = act_pred // (3 ** 2) # [B, ]
        # y_idx_pred = (act_pred - (3 ** 2) * x_idx_pred) // 3
        # z_idx_pred = act_pred - (3 ** 2) * x_idx_pred - 3 * y_idx_pred

    XAI_DIR = os.path.expanduser(f"~/xai_results/sac/{CFG.TASK}/{CFG.EXP_NAME}/")
    # XAI
    if CFG.OBS_TYPE == 'vision':
        model.step()
        if model.global_step % CFG.XAI_INTERVAL == 0:
            obs = obs_raw
            if CFG.GCAM:
                # Grad-CAM
                result_images = None
                for axis in range(3):
                    
                    # update GradCAM
                    state = (obs * 255.0).float() #.permute(0,3,1,2)
                    action = model.gcam.forward(torch.unsqueeze(state[0], 0))
                    boundary = torch.tensor([-0.3, 0.0, 0.3], device=obs.device)

                    # (1) Get state image
                    state = state[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    state = cv2.resize(state, (150, 150), interpolation=cv2.INTER_LINEAR)
                    
                    # Get Grad-CAM image (3X3)
                    # for target_layer in target_layers:   
                    target_layer = "_convs.1"
                
                    # (2) Get regions for each layer of model
                    model.gcam.backward(axis)
                    regions = model.gcam.generate(target_layer)
                    regions = regions.detach().cpu().numpy()
                    regions = np.squeeze(regions) * 255
                    regions = np.transpose(regions)
                    
                    # Resizing the heatmap of region
                    regions = cv2.applyColorMap(regions.astype(np.uint8), cv2.COLORMAP_JET)
                    regions = cv2.resize(regions, (150, 150), interpolation=cv2.INTER_LINEAR)
                    regions = cv2.cvtColor(regions, cv2.COLOR_RGB2BGR)

                    # (3) Overlay the state & region.
                    overlay = cv2.addWeighted(state, 1.0, regions, 0.5, 0)
                    
                    # Concate (1)~(3)
                    result = np.hstack([state, regions, overlay])
                    result_images = (
                        result
                        if result_images is None
                        else np.vstack([result_images, result])
                    )
                result_images = cv2.copyMakeBorder(result_images,30,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
                # Show action on result image
                cv2.putText(
                    img=result_images,
                    text="action : {:0.3f},{:0.3f},{:0.3f}".format(action[0],action[1],action[2]),
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                )
                xai_dir = XAI_DIR + '/gcam'
                Path(xai_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(xai_dir + '/{}_gcam.png'.format(model.global_step), cv2.cvtColor(result_images, cv2.COLOR_RGB2BGR))
                
            if CFG.SMAP:
                # Saliency map.
                xai_dir = XAI_DIR + '/smap'
                Path(xai_dir).mkdir(parents=True, exist_ok=True) # saliency_map_dir = make_saliency_dir(xai_dir)
                
                # update GradCAM
                state = (obs * 255.0).float() #.permute(0,3,1,2)

                with FreezeParameters(model.parameters()):
                    saliency_map, scores = compute_saliency_maps(
                        state = state[0].unsqueeze(0),
                        model = model,
                        device = policy.device
                        )
                        
                state = state[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8) # .astype(np.uint8)
                # state = state[0].detach().cpu().numpy() # .astype(np.uint8)
                state = cv2.resize(state, (150, 150), interpolation=cv2.INTER_LINEAR)
                state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
                result_images = None
                for smap in saliency_map:
                    smap = smap.detach().cpu().numpy()

                    # min_val = smap.min()
                    # max_val = smap.max()
                    # smap = smap - min_val
                    # smap = smap / (max_val - min_val) * 255
                    smap = heatmap(smap.transpose(1, 2, 0)) * 255.0
                    # smap = cv2.applyColorMap(smap[0].astype(np.uint8), cv2.COLORMAP_HOT)
                    smap = cv2.resize(smap, (150, 150), interpolation=cv2.INTER_LINEAR)
                    overlay = cv2.addWeighted(state, 1.0, smap, 0.5, 0)
                    result = np.hstack([state, smap, overlay])
                    result_images = (
                        result
                        if result_images is None
                        else np.vstack([result_images, result])
                    )
                # Show action on result image
                result_images = cv2.copyMakeBorder(result_images,30,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
                scores = scores.detach().cpu().numpy()
                cv2.putText(
                    img=result_images,
                    text="action : {:0.3f},{:0.3f},{:0.3f}".format(scores[0],scores[1],scores[2]),
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=2,
                )
                cv2.imwrite(xai_dir + f'/test_step{model.global_step}.png', result_images)

            # TODO (Chmin): we should only parse the actor part of the model.
            if CFG.LRP:
                xai_dir = XAI_DIR + '/lrp'
                Path(xai_dir).mkdir(parents=True, exist_ok=True) # saliency_map_dir = make_saliency_dir(xai_dir)

                model.eval()
                # TODO (chmin): check if tanh could be parsed.
                act_infer_list = [model._convs, model._logits, SlimFlatten(), model.action_model]
                lrp_model = convert_vision(act_infer_list).to(obs.device)
                # _ = lrp_model(obs.permute(0, 3, 1, 2)) # [B, 2 * A]
                _ = lrp_model(obs) # [B, 2 * A]
                
                input = obs.contiguous().detach().cpu().numpy()[0][None][0].transpose(1, 2, 0)
                #plot for each explanation
                result_images = None
                action = np.zeros(3)
                for axis in range(3):
                    result = cv2.resize(input * 255, (150, 150), interpolation=cv2.INTER_LINEAR)
                    for _, (rule, pattern) in enumerate(explanations):
                        # attr, action[axis] = compute_and_plot_explanation(lrp_model, obs=obs[0][None].permute(0, 3, 1, 2), rule=rule, axis = axis, patterns=pattern)
                        attr, action[axis] = compute_and_plot_explanation(lrp_model, obs=obs[0][None], rule=rule, axis = axis, patterns=pattern)
                        attr = attr.permute(0, 2, 3, 1)
                        attr = heatmap(attr) * 255
                        attr = cv2.resize(attr[0], (150, 150), interpolation=cv2.INTER_LINEAR)
                        if result_images is None:
                            if result.shape[0] == 150:
                                result = cv2.copyMakeBorder(result,30,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])                
                            attr = cv2.copyMakeBorder(attr,30,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])                
                            cv2.putText(
                                img=attr,
                                text=rule,
                                org=(20, 20),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1,
                                color=(255, 0, 0),
                                thickness=1,
                            )
                        result = np.hstack([result, attr])
                    result_images = (
                        result
                        if result_images is None
                        else np.vstack([result_images, result])
                    )
                # Show action on result image
                result_images = cv2.copyMakeBorder(result_images,30,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
                
                cv2.putText(
                    img=result_images,
                    text="action : {:0.3f},{:0.3f},{:0.3f}".format(action[0],action[1],action[2]),
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                )
                cv2.imwrite(xai_dir + f'/test_step{model.global_step}.png', cv2.cvtColor(result_images, cv2.COLOR_RGB2BGR))
                model.train()

    elif CFG.OBS_TYPE == 'state':
        pass
    else:
        raise ValueError
    
    # Save for stats function.
    policy.q_t = q_t
    policy.policy_t = policy_t
    policy.log_pis_t = log_pis_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.log_alpha_value = model.log_alpha
    policy.alpha_value = alpha
    policy.target_entropy = model.target_entropy
    policy.action_dist_norm = action_dist_norm
    # policy.acttion_discretize_loss = act_disc_loss


    # visualization
    if (policy.global_timestep % 1000 == 0 and hasattr(model, 'episodic_step')
        and hasattr(model, 'vis_episode')):
        policy.vis_episode = model.vis_episode
        # TODO (chmin): process as video (gif) here.



    # Return all loss terms corresponding to our optimizers.
    return tuple([policy.actor_loss] + policy.critic_loss +
                 [policy.alpha_loss])


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    if policy.global_timestep % CFG.VIS_EVERY == 0 and hasattr(policy, 'vis_episode'):
        episode_gif = policy.vis_episode #.permute(0, 1, 4, 2, 3)
    else:
        episode_gif = None


    return {
        "td_error": policy.td_error,
        "mean_td_error": torch.mean(policy.td_error),
        "actor_loss": torch.mean(policy.actor_loss),
        "critic_loss": torch.mean(torch.stack(policy.critic_loss)),
        "alpha_loss": torch.mean(policy.alpha_loss),
        "alpha_value": torch.mean(policy.alpha_value),
        "log_alpha_value": torch.mean(policy.log_alpha_value),
        "target_entropy": policy.target_entropy,
        "policy_t": torch.mean(policy.policy_t),
        "mean_q": torch.mean(policy.q_t),
        "max_q": torch.max(policy.q_t),
        "min_q": torch.min(policy.q_t),
        "action_dist_norm" : policy.action_dist_norm,
        "episode_gif": episode_gif
    }


def optimizer_fn(policy: Policy, config: TrainerConfigDict) -> \
        Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.

    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy (Policy): The policy object to be trained.
        config (TrainerConfigDict): The Trainer's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=policy.model.q_variables()[:critic_split],
            lr=config["optimization"]["critic_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables()[critic_split:],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
            ))
    policy.alpha_optim = torch.optim.Adam(
        params=[policy.model.log_alpha],
        lr=config["optimization"]["entropy_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    return tuple([policy.actor_optim] + policy.critic_optims +
                 [policy.alpha_optim])


class ComputeTDErrorMixin:
    """Mixin class calculating TD-error (part of critic loss) per batch item.

    - Adds `policy.compute_td_error()` method for TD-error calculation from a
      batch of observations/actions/rewards/etc..
    """

    def __init__(self):
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_t,
                SampleBatch.ACTIONS: act_t,
                SampleBatch.REWARDS: rew_t,
                SampleBatch.NEXT_OBS: obs_tp1,
                SampleBatch.DONES: done_mask,
                PRIO_WEIGHTS: importance_weights,
            })
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            actor_critic_loss(self, self.model, None, input_dict)

            # `self.td_error` is set within actor_critic_loss call. Return
            # its updated value here.
            return self.td_error

        # Assign the method to policy (self) for later usage.
        self.compute_td_error = compute_td_error


class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")
        model_state_dict = self.model.state_dict()
        # Support partial (soft) synching.
        # If tau == 1.0: Full sync from Q-model to target Q-model.
        if tau != 1.0:
            target_state_dict = self.target_model.state_dict()
            model_state_dict = {
                k: tau * model_state_dict[k] + (1 - tau) * v
                for k, v in target_state_dict.items()
            }
        self.target_model.load_state_dict(model_state_dict)


def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      config: TrainerConfigDict) -> None:
    """Call mixin classes' constructors after Policy initialization.

    - Moves the target model(s) to the GPU, if necessary.
    - Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).
    - Also adds the `update_target` method to the given policy.
    Calling `update_target` updates all target Q-networks' weights from their
    respective "main" Q-metworks, based on tau (smooth, partial updating).

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    policy.target_model = policy.target_model.to(policy.device)
    policy.model.log_alpha = policy.model.log_alpha.to(policy.device)
    policy.model.target_entropy = policy.model.target_entropy.to(policy.device)
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
SACTorchPolicy = build_torch_policy(
    name="SACTorchPolicy",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
