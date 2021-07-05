from typing import Optional, List, Dict, Tuple

import gym
from gym.spaces.box import Box
from numpy.lib.arraysetops import isin
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override, force_list
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class RNNSACTorchModel(SACTorchModel):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: Optional[int],
                 model_config: ModelConfigDict,
                 name: str,
                 actor_hidden_activation: str = "relu",
                 actor_hiddens: Tuple[int] = (256, 256),
                 critic_hidden_activation: str = "relu",
                 critic_hiddens: Tuple[int] = (256, 256),
                 twin_q: bool = False,
                 initial_alpha: float = 1.0,
                 target_entropy: Optional[float] = None,
                 global_step: int = 0):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            actor_hidden_activation=actor_hidden_activation,
            actor_hiddens=actor_hiddens,
            critic_hidden_activation=critic_hidden_activation,
            critic_hiddens=critic_hiddens,
            twin_q=twin_q,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy)

        self.use_prev_action = (model_config["lstm_use_prev_action"]
                                or model_config['policy_model']["lstm_use_prev_action"]
                                or model_config['Q_model']["lstm_use_prev_action"])

        self.use_prev_reward = (model_config["lstm_use_prev_reward"]
                                or model_config['policy_model']["lstm_use_prev_reward"]
                                or model_config['Q_model']["lstm_use_prev_reward"])

        if self.use_prev_action:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = \
                ViewRequirement(SampleBatch.ACTIONS, space=self.action_space,
                                shift=-1)
        if self.use_prev_reward:
            self.view_requirements[SampleBatch.PREV_REWARDS] = \
                ViewRequirement(SampleBatch.REWARDS, shift=-1)

    # @override(SACTorchModel)
    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """The common (Q-net and policy-net) forward pass.
        NOTE: It is not(!) recommended to override this method as it would
        introduce a shared pre-network, which would be updated by both
        actor- and critic optimizers.
        For rnn support remove input_dict filter and pass state and seq_lens
        """
        model_out = {"obs": input_dict[SampleBatch.OBS]}

        if self.use_prev_action:
            model_out["prev_actions"] = input_dict[SampleBatch.PREV_ACTIONS]
        if self.use_prev_reward:
            model_out["prev_rewards"] = input_dict[SampleBatch.PREV_REWARDS]

        return model_out, state

    # @override(SACTorchModel)
    # def _get_q_value(self, model_out: TensorType, actions, net,
    #                  state_in: List[TensorType],
    #                  seq_lens: TensorType) -> (TensorType, List[TensorType]):
    #     # Continuous case -> concat actions to model_out.
    #     if actions is not None:
    #         if self.concat_obs_and_actions:
    #             model_out[SampleBatch.OBS] = \
    #                 torch.cat([model_out[SampleBatch.OBS], actions], dim=-1)
    #         else:
    #             model_out[SampleBatch.OBS] = \
    #                 force_list(model_out[SampleBatch.OBS]) + [actions]

    #     # Switch on training mode (when getting Q-values, we are usually in
    #     # training).
    #     model_out["is_training"] = True

    #     out, state_out = net(model_out, state_in, seq_lens)
    #     return out, state_out

    @override(SACTorchModel)
    def get_q_values(self,
                     model_out: TensorType,
                     state_in: List[TensorType],
                     seq_lens: TensorType,
                     actions: Optional[TensorType] = None) -> TensorType:

        # return self._get_q_value(model_out, actions, self.q_net, state_in,
        #                          seq_lens)
        if isinstance(self.q_net.obs_space, Box):
            if isinstance(model_out, (list, tuple)):
                model_out = torch.cat(model_out, dim=-1)

            elif isinstance(model_out, dict):
                model_out = torch.cat(list(model_out.values()), dim=-1)

        elif isinstance(model_out, dict):
            model_out = list(model_out.values())

        if actions is not None:
            input_dict = {'obs': torch.cat([model_out, actions], dim=-1)}

        else:
            input_dict = {'obs': model_out}

        # 
        input_dict["is_training"] = True
        out, _ = self.q_net(input_dict, [], None)

        return out

    @override(SACTorchModel)
    def get_twin_q_values(self,
                          model_out: TensorType,
                          state_in: List[TensorType],
                          seq_lens: TensorType,
                          actions: Optional[TensorType] = None) -> TensorType:
        # TODO (chmin): this should be implemented later.

        if isinstance(self.twin_q_net.obs_space, Box):
            if isinstance(model_out, (list, tuple)):
                model_out = torch.cat(model_out, dim=-1)

            elif isinstance(model_out, dict):
                model_out = torch.cat(list(model_out.values()), dim=-1)

        elif isinstance(model_out, dict):
            model_out = list(model_out.values())

        if actions is not None:
            input_dict = {'obs': torch.cat([model_out, actions], dim=-1)}

        else:
            input_dict = {'obs': model_out}

        # 
        input_dict["is_training"] = True
        out, _ = self.twin_q_net(input_dict, [], None)

        return out


    @override(SACTorchModel)
    def get_policy_output(
            self, model_out: TensorType, state_in: List[TensorType],
            seq_lens: TensorType) -> (TensorType, List[TensorType]):
        # TODO (chmin): should adapt to RNN states.
        return self.action_model(model_out, state_in, seq_lens)

    @override(ModelV2)
    def get_initial_state(self):
        # TODO (chmin): implement a method that returns initial RNN states
        policy_initial_state = self.action_model.get_initial_state()
        q_initial_state = self.q_net.get_initial_state()
        if self.twin_q_net:
            q_initial_state *= 2
        return policy_initial_state + q_initial_state

    def select_state(self, state_batch: List[TensorType],
                     net: List[str]) -> Dict[str, List[TensorType]]:
        assert all([n in ["policy", "q", "twin_q"] for n in net]), \
            "Selected state must be either for policy, q or twin_q network"
        policy_state_len = len(self.action_model.get_initial_state())
        q_state_len = len(self.q_net.get_initial_state())

        selected_state = {}
        for n in net:
            if n == "policy":
                selected_state[n] = state_batch[:policy_state_len]
            elif n == "q":
                selected_state[n] = state_batch[policy_state_len:
                                                policy_state_len + q_state_len]
            elif n == "twin_q":
                if self.twin_q_net:
                    selected_state[n] = state_batch[policy_state_len +
                                                    q_state_len:]
                else:
                    selected_state[n] = []
        return selected_state