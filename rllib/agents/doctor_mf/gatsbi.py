import logging

import random
import numpy as np
import torch

from ray.rllib.agents import with_common_config
from gatsbi_rl.rllib_agent.gatsbi_torch_policy import GATSBITorchPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, \
    LEARNER_INFO, _get_shared_metrics, _get_global_vars
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.evaluation.metrics import collect_metrics
# from ray.rllib.agents.dreamer.dreamer_model import DreamerModel
from gatsbi_rl.rllib_agent.gatsbi_model import GATSBIModel
# from ray.rllib.agents.gatsbi.gatsbi_model import GATSBIModel
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.utils.typing import SampleBatchType
# offline datareader
from ray.rllib.offline import JsonReader
from gatsbi_rl.gatsbi.arch import ARCH
from .utils import scale_action

import os
from ray.rllib.utils.schedules import PiecewiseSchedule
from gatsbi_rl.gatsbi.utils import bcolors

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
# TODO (chmin): the config should be induced from arch.py.
DEFAULT_CONFIG = with_common_config({
    # GATSBI Model LR
    "td_model_lr": 6e-4,
    # Actor LR
    "actor_lr": 8e-5,
    # Critic LR
    "critic_lr": 8e-5,
    # Grad Clipping
    "grad_clip": 1.0,
    # Discount
    "discount": 0.99,
    # Lambda
    "lambda": 0.95,
    # Training iterations per data collection from real env
    "gatsbi_train_iters": 100,
    # Horizon for Enviornment (1000 for Mujoco/DMC) #? maybe the episode length?
    "horizon": 100,  #! debug
    # Number of episodes to sample for Loss Calculation
    "batch_size": 2,
    # Length of each episode to sample for Lofgatss Calculation
    "batch_length": 10,

    "lr_schedule": None,
    # Imagination Horizon for Training Actor and Critic # prediction horizon.
    "imagine_horizon": 5,
    # Free Nats
    "free_nats": 3.0, # lower divergence bound for the model loss.
    # KL Coeff for the Model Loss
    "kl_coeff": 1.0,
    # Distributed GATSBI not implemented yet
    "num_workers": 0,
    # Prefill Timesteps
    "prefill_timesteps": 10000, #! debug

    "eval_episodes": 50, # whole episode
    "eval_length": 300, # whole episode
    "max_buffer_episodes": 500, # max capacity of replay buffer (1000 episode requires 30 GB RAM)
    "max_demo_episodes": 500, # max capacity of replay buffer (1000 episode requires 30 GB RAM)
    # This should be kept at 1 to preserve sample efficiency
    "num_envs_per_worker": 1,
    # Exploration Gaussian
    "explore_noise": 0.3,
    # Batch mode
    "batch_mode": "complete_episodes",
    # Custom Model
    # TODO (chmin): synchronize the model params with those in arch.py
    "gatsbi_model": {
        "custom_model": GATSBIModel,
        "global_step": 0, # global_step set by the user on model creation
    },
    # TODO (chmin): there should be no action repeat!
    "env_config": {
        # Repeats action send by policy for frame_skip times in env
        "frame_skip": 1,
    }
})
# __sphinx_doc_end__
# yapf: enable


class EpisodicBuffer(object):
    def __init__(self, max_length: int = 1000, length: int = 50):
        """Data structure that stores episodes and samples chunks
        of size length from episodes
        Args:
            max_length: Maximum episodes it can store
            length: Episode chunking lengh in sample()
        """

        # Stores all episodes into a list: List[SampleBatchType]
        self.episodes = []
        self.max_length = max_length
        self.timesteps = 0
        self.length = length

    def add(self, batch: SampleBatchType):
        """Splits a SampleBatch into episodes and adds episodes
        to the episode buffer

        Args:
            batch: SampleBatch to be added
        """
        # TODO (chmin): add whole episode for visualization (eval).
        self.timesteps += batch.count # count from dtype Union
        episodes = batch.split_by_episode()

        for i, e in enumerate(episodes):
            episodes[i] = self.preprocess_episode(e)
        self.episodes.extend(episodes)

        if len(self.episodes) > self.max_length:
            delta = len(self.episodes) - self.max_length
            # Drop oldest episodes
            self.episodes = self.episodes[delta:]

    def preprocess_episode(self, episode: SampleBatchType):
        """Batch format should be in the form of (s_t, a_(t-1), r_(t-1))
        When t=0, the resetted obs is paired with action and reward of 0.
        Args:
            episode: SampleBatch representing an episode
        """
        if 'bair_push' in ARCH.REAL_WORLD and episode.count != ARCH.HORIZON:
            batch_obs = episode["obs"]
            action = episode["actions"]
            act_shape = action.shape
            act_reset = np.array([0.0] * act_shape[-1])[None]
            batch_action = action # a_{t-1:t+H}
            batch_next_action = np.concatenate([action[1:], act_reset], axis=0)
            ep_len = batch_obs.shape[0]
            batch_rew = np.zeros((ep_len, 1), dtype=np.float32)
        elif 'robonet' in ARCH.REAL_WORLD and episode.count != ARCH.HORIZON:
            batch_obs = episode["obs"].transpose(0, 3, 1, 2)
            action = episode["actions"]
            act_shape = action.shape
            act_reset = np.array([0.0] * act_shape[-1])[None]
            batch_action = np.concatenate([act_reset, action[:-1]], axis=0) # a_{t-1:t+H}
            batch_next_action = action # a_{t:t+H+1}
            ep_len = batch_obs.shape[0]
            batch_rew = np.zeros((ep_len, 1), dtype=np.float32)
        elif 'sketchy' in ARCH.REAL_WORLD and episode.count != ARCH.HORIZON:
            batch_obs = episode["obs"].transpose(0, 3, 1, 2)
            action = episode["actions"]
            act_shape = action.shape
            act_reset = np.array([0.0] * act_shape[-1])[None]
            batch_action = np.concatenate([act_reset, action[:-1]], axis=0) # a_{t-1:t+H}
            batch_next_action = action # a_{t:t+H+1}
            ep_len = batch_obs.shape[0]
            batch_rew = np.zeros((ep_len, 1), dtype=np.float32)
        else:
            obs = episode["obs"]
            new_obs = episode["new_obs"]
            action = episode["actions"]
            reward = episode["rewards"]

            act_shape = action.shape
            act_reset = np.array([0.0] * act_shape[-1])[None]
            rew_reset = np.array(0.0)[None]
            obs_end = np.array(new_obs[act_shape[0] - 1])[None]

            batch_obs = np.concatenate([obs, obs_end], axis=0)
            # TODO (chmin): tentative change of action spaces
            # batch_action = np.concatenate([act_reset, action], axis=0)
            # it will be sliced
            if act_shape[-1] == 8: # demo -> dim as 7
                batch_action = np.concatenate([act_reset[:, :-1], action[:, :-1]], axis=0) # a_{t-1:t+H}
                batch_next_action = np.concatenate([action[:, :-1], act_reset[:, :-1]], axis=0) # a_{t:t+H+1}
            else:
                batch_action = np.concatenate([act_reset, action], axis=0)
                batch_next_action = np.concatenate([action, act_reset], axis=0)
            batch_rew = np.concatenate([rew_reset, reward], axis=0)

        new_batch = {
            "obs": batch_obs,
            "rewards": batch_rew,
            "actions": batch_action,
            "next_actions": batch_next_action,
        }
        return SampleBatch(new_batch)

    def sample(self, batch_size: int, full_episode=False):
        """Samples [batch_size, length] from the list of episodes
            length -> episode chunking length.
        Args:
            batch_size: batch_size to be sampled
            full_episode:
        """
        episodes_buffer = []
        min_len = 0 # maximum length of episode in evaluation batch
        while len(episodes_buffer) < batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]
            min_len = episode.count if episode.count < min_len or min_len == 0 else min_len

            if episode.count < self.length and not full_episode:
                # if the length of the chunk is less than specified one,
                # discard the sample. 
                continue
            if not full_episode:
                available = episode.count - self.length
                # index in a raepisodedom.randint(0, available))
                index = int(random.randint(0, available))
                episodes_buffer.append(episode.slice(index, index + self.length))
            else:
                episodes_buffer.append(episode)
        # reformulate the dictionary.

        batch = {}
        for k in episodes_buffer[0].keys():
            if not full_episode:
                batch[k] = np.stack([e[k] for e in episodes_buffer], axis=0)
            else:
                batch[k] = np.stack([e.slice(0, min_len)[k] for e in episodes_buffer], axis=0)
        return SampleBatch(batch)


def total_sampled_timesteps(worker):
    return worker.policy_map[DEFAULT_POLICY_ID].global_timestep


class GATSBIIteration:
    """ The main iterator for learing the GATSBI model.
    """
    def __init__(self, worker, episode_buffer, gatsbi_train_iters, batch_size,
                 act_repeat, eval_buffer=None, demo_buffer=None, global_step=0):
        self.worker = worker
        self.episode_buffer = episode_buffer
        self.prefilled_steps = self.episode_buffer.timesteps
        self.gatsbi_train_iters = gatsbi_train_iters
        self.repeat = act_repeat
        self.batch_size = batch_size
        self.total_count = global_step
        print(bcolors.OKGREEN + "Iteration is intialized as {0}".format(global_step) + bcolors.ENDC)
        if ARCH.REAL_WORLD != 'false': # real world sample learning.
            DEMO_ROOT = '~/rlbench_data/' + ARCH.REAL_WORLD
        else: # simulated learning
            DEMO_ROOT = '~/rlbench_data/' + ARCH.TASK_NAME + '_demo'
        print(bcolors.FAIL + "Loading demo from " + DEMO_ROOT + bcolors.ENDC)
        # put the whole samples into the buffer
        self.eval_buffer = eval_buffer
        print("Start storing eval episodes on eval buffer.")
        if ARCH.PRETRAIN_DEMO:
            self.eval_reader = JsonReader(os.path.expanduser(DEMO_ROOT))
            self.demo_reader = JsonReader(os.path.expanduser(DEMO_ROOT))
            while True:
                try: # TODO (chmin): demo should have optimized format.
                    eval_sample = self.eval_reader.next()
                    eval_sample = self.process_demo(eval_sample)
                    self.eval_buffer.add(eval_sample)
                    ep_len = len(self.eval_buffer.episodes)
                    if ep_len == self.eval_buffer.max_length:
                        print("Stored up to full capacity of eval buffer.")
                        break
                except:
                    # continue
                    print("Added total {0} demo episodes".format(len(
                        self.eval_buffer.episodes)))
                    break
            # if ARCH.PRETRAIN_DEMO:
            self.demo_buffer = demo_buffer
            print("Start storing demo data on demo buffer.")
            while True:
                try: # TODO (chmin): demo should have optimized format.
                    demo_sample = self.demo_reader.next()
                    demo_sample = self.process_demo(demo_sample)
                    self.demo_buffer.add(demo_sample)
                    ep_len = len(self.demo_buffer.episodes)
                    if ep_len % 100 == 0:
                        print("Currently {0} episodes have been stored.".format(ep_len))
                    if ep_len == self.demo_buffer.max_length:
                        print("Stored upto full capacity of demo buffer.")
                        break
                except:
                    # continue
                    print("Added total {0} demo episodes".format(len(
                        self.demo_buffer.episodes)))
                    break

    def __call__(self, samples):
        # TODO (chmin): train with demo here.
        # 'samples' are from the direct interaction, thus having raw actions.
        # GATSBI Training Loop
        for n in range(self.gatsbi_train_iters): # This is where the training iter starts.
            self.total_count += 1
            if ARCH.PRETRAIN_DEMO and self.total_count <= ARCH.PRETRAIN_GATSBI_UNTIL:
                batch = self.demo_buffer.sample(self.batch_size)
            else:
                if self.total_count <= ARCH.JOINT_TRAIN_GATSBI_START:
                    if torch.rand(1) >= 0.5:
                        batch = self.demo_buffer.sample(self.batch_size)
                    else:
                        batch = self.episode_buffer.sample(self.batch_size) # []
                else:
                    if ARCH.VISUALIZE:
                        batch = self.demo_buffer.sample(self.batch_size)
                    else:
                        # if torch.rand(1) >= 0.5:
                        #     batch = self.demo_buffer.sample(self.batch_size)
                        # else:
                        batch = self.episode_buffer.sample(self.batch_size) # []
            if self.total_count <= ARCH.PRETRAIN_GATSBI_UNTIL:
                eval_batch = self.eval_buffer.sample(ARCH.NUM_GEN, full_episode=True) # []
            else:
                if torch.rand(1) >= 0.5:
                    eval_batch = self.eval_buffer.sample(ARCH.NUM_GEN, full_episode=True) # []
                else:
                    eval_batch = self.episode_buffer.sample(ARCH.NUM_GEN, full_episode=True) # []
            # if n == self.gatsbi_train_iters - 1:
            if self.total_count % ARCH.VIS_EVERY == 0:
                batch["log_gif"] = True
                batch["eval_obs"] = eval_batch["obs"] # full episode sequence
                batch["eval_actions"] = eval_batch["actions"] # full episode sequence
                batch["eval_rewards"] = eval_batch["rewards"] # full episode sequence

            # NOTE that we sample the training batch here!
            # visualization should refer to it.
            fetches = self.worker.learn_on_batch(batch)
            # get_global_var is based upon "metrics.counters[STEPS_SAMPLED_COUNTER]"
            # set_global_var parses the value to lr_scheduler.
            # if self.total_count < ARCH.PRETRAIN_GATSBI_UNTIL: #* global-timestep is episodic interaction.
            metrics = _get_shared_metrics() # decay learning rate.
            metrics.counters[STEPS_SAMPLED_COUNTER] = self.total_count #  - self.prefilled_steps

            # decay learning rate
            self.worker.set_global_vars(_get_global_vars())

        # TODO (chmin): apply lr scheduling here.

        policy_fetches = self.policy_stats(fetches)
        if "log_gif" in policy_fetches:
            track_grid, track_gifs, gen_grid, gen_gifs = policy_fetches["log_gif"] # numpy objects
            policy_fetches["track_gifs"], policy_fetches['gen_gifs'] = self.postprocess_gif(track_gifs, gen_gifs)
            policy_fetches["track_grid"], policy_fetches["gen_grid"] = track_grid, gen_grid
            del policy_fetches['log_gif']
        # TODO (chmin): note that the visualization was done after each iteration.a
        # Metrics Calculation

        metrics.info[LEARNER_INFO] = fetches

        # metrics.counters[STEPS_SAMPLED_COUNTER] *= self.repeat
        res = collect_metrics(local_worker=self.worker)
        res["info"] = metrics.info
        res["info"].update(metrics.counters)

        res["timesteps_total"] = metrics.counters[STEPS_SAMPLED_COUNTER]
        samples.data['actions'] = scale_action(samples.data['actions'])
        if (self.total_count >= ARCH.PRETRAIN_GATSBI_UNTIL and ARCH.PRETRAIN_DEMO) or not ARCH.PRETRAIN_DEMO: #* increase episodic count.
            # TODO (chmin): scale the policy action output.
            # by doing so, all 'batch' data has scaled actions. (demos are already scaled)
            self.eval_buffer.add(samples)
        self.episode_buffer.add(samples) #* global_step should not increase by this.
        
        return res

    def process_demo(self, demo_sample):
        """ Process rlbench demo data into SampleBatch of ray.
        """
        demo_sample.count -= ARCH.SLICE_DEMO_FROM
        demo_sample.data = dict((k, v[ARCH.SLICE_DEMO_FROM:]) for k, v in demo_sample.data.items())
        if 'bair_push' in ARCH.REAL_WORLD: # real world sample learning.
            demo_sample['obs'] = demo_sample['obs'].astype(np.float32) / 255.
            ee_pos = demo_sample.data['ee_pos'] # [30, 3]
            ee_pos_prev = ee_pos[:-1] # [29, 3]
            ee_pos_next = ee_pos[1:] # [29, 3]
            ee_pos_diff = ee_pos_next - ee_pos_prev # [29, 3]
            ee_pos_diff_init = np.zeros((1, 3), dtype=np.float32)
            action = np.concatenate([ee_pos_diff_init, ee_pos_diff], axis=0)
            demo_sample.data['actions'] = action
        elif 'robonet' in  ARCH.REAL_WORLD: # real world sample learning.
            demo_sample['obs'] = demo_sample['obs'].astype(np.float32) / 255.

        elif 'sketchy' in  ARCH.REAL_WORLD: # real world sample learning.
            demo_sample['obs'] = demo_sample['obs'].astype(np.float32) / 255.
            demo_sample['obs'] = demo_sample['obs'].squeeze(1)
            # in sketchy datset, agent grasps the objects. 
            demo_sample['actions'] = np.concatenate([demo_sample['actions'], demo_sample['grasp']], axis=-1)
            demo_sample['actions'] = demo_sample['actions'].squeeze(1)
        # else: # simulated dataset
        #     demo_sample.data['actions'][0] = [0.0] * 8
        #     demo_sample.data['actions'] = np.array(list(demo_sample.data['actions']))
        return demo_sample


    def postprocess_gif(self, track_gifs, gen_gifs):
        # gifs has the shape
        track_gifs = np.clip(255 * track_gifs, 0, 255).astype(np.uint8)
        gen_gifs = np.clip(255 * gen_gifs, 0, 255).astype(np.uint8)
        B, T, C, H, W = track_gifs.shape
        return track_gifs, gen_gifs

    def policy_stats(self, fetches):
        return fetches["default_policy"]["learner_stats"]


def execution_plan(workers, config):
    # Special Replay Buffer for GATSBI agent
    episode_buffer = EpisodicBuffer(max_length=config["max_buffer_episodes"],
             length=config["batch_length"])

    # TODO (chmin): Note that episodes here are only used for evaluating GATSBI pretraining.
    eval_buffer = EpisodicBuffer(max_length=config["eval_episodes"],
            length=config["eval_length"])

    demo_buffer = None
    if ARCH.PRETRAIN_DEMO:
        demo_buffer = EpisodicBuffer(max_length=config["max_demo_episodes"],
            length=config["batch_length"])

    local_worker = workers.local_worker()

    # Prefill episode buffer with initial exploration (uniform sampling)
    while total_sampled_timesteps(local_worker) < config["prefill_timesteps"]:
        samples = local_worker.sample() #! This executes a single episode. May take a while.
        # process action.
        # TODO (chmin): prefill actions should also be scaled!
        samples.data['actions'] = scale_action(samples.data['actions'])
        episode_buffer.add(samples)
        if not ARCH.PRETRAIN_DEMO:
            eval_buffer.add(samples)

    print(bcolors.WARNING + "Initialized the agent with {} prefill data. \
        ".format(episode_buffer.timesteps) + bcolors.ENDC)

    batch_size = config["batch_size"]
    gatsbi_train_iters = config["gatsbi_train_iters"]
    act_repeat = config["action_repeat"]

    rollouts = ParallelRollouts(workers)
    rollouts = rollouts.for_each(
        GATSBIIteration(local_worker, episode_buffer, gatsbi_train_iters,
            batch_size, act_repeat, eval_buffer=eval_buffer, demo_buffer=demo_buffer,
            global_step=config["gatsbi_model"]["global_step"]
            ))
    return rollouts


def get_policy_class(config):
    return GATSBITorchPolicy


def validate_config(config):
    config["action_repeat"] = config["env_config"]["frame_skip"]
    if config["framework"] != "torch":
        raise ValueError("GATSBI not supported in Tensorflow yet!")
    if config["batch_mode"] != "complete_episodes":
        raise ValueError("truncate_episodes not supported")
    if config["num_workers"] != 0:
        raise ValueError("Distributed GATSBI not supported yet!")
    if config["clip_actions"]:
        raise ValueError("Clipping is done inherently via policy tanh!")
    if config["action_repeat"] > 1:
        config["horizon"] = config["horizon"] / config["action_repeat"]


GATSBITrainer = build_trainer(
    name="GATSBI",
    default_config=DEFAULT_CONFIG,
    default_policy=GATSBITorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    validate_config=validate_config)
