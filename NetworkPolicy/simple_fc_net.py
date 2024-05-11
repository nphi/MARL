import logging
from typing import Callable

import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CustomFCNet(TorchModelV2, nn.Module):
    """Basic Fully Connected Neural Network Implementation"""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # This is tracked only for metrics
        self.action_calls = 0
        self.value_calls = 0

        self.activations = {}
        self.hooks = []

        self.last_observation = None  # We track this, so it can be re-used during the value-function call.
        input_layer_size = int(np.product(obs_space.shape))  # flattened observation shape

        # Policy NN layers
        # Describes a 200 -> [200,256] -> [256, 256] -> [256, 4] -> 4
        self.policy_fc1 = SlimFC(
            in_size=input_layer_size,
            out_size=256,
            activation_fn='tanh',
            initializer=normc_initializer(1.0),
        )
        self.policy_fc2 = SlimFC(
            in_size=256,
            out_size=256,
            activation_fn='tanh',
            initializer=normc_initializer(1.0),
        )
        self.policy_output_fc = SlimFC(
            in_size=256,
            out_size=num_outputs,
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )

        # Value Function NN layers
        # Describes a 200 -> [200,256] -> [256, 256] -> [256, 1] -> 1
        self.value_fc1 = SlimFC(
            in_size=input_layer_size,
            out_size=256,
            activation_fn='tanh',
            initializer=normc_initializer(1.0),
        )
        self.value_fc2 = SlimFC(
            in_size=256,
            out_size=256,
            activation_fn='tanh',
            initializer=normc_initializer(1.0),
        )
        self.value_output_fc = SlimFC(
            in_size=256,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )

    def register_activation_hooks(self):
        """
        Adds hooks to save activations from all neural network layers in this class.
        For more details on how this works, see:
        https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        :return:
        """
        layer_names = set([name.split('.')[0] for name, _ in self.named_parameters()])

        # This yields a method that adds an activation to our dictionary.
        def save_activations_for(name) -> Callable:
            def save(model, input, output):
                self.activations[name].append(output.cpu().detach())

            return save

        for name in layer_names:
            layer = getattr(self, name)
            self.activations[name] = [] # Create activations list
            # We access the internal model of the RLlib module here. Not great, but it's the only thing that works.
            hook = layer._model.register_forward_hook(save_activations_for(name))
            self.hooks.append(hook)

    def deregister_activation_hooks(self):
        """
        Removes stored activation hooks
        :return:
        """
        for hook in self.hooks:
            hook.remove()

    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """
        Method that calls policy neural network to get back scores for each possible action.
        :param input_dict: contains dictionary of last observation and more.
        :param state:
        :param seq_lens:
        :return:
        """
        self.action_calls += 1
        obs = input_dict["obs_flat"].float()  # Flattened observation: B x 200 tensor
        self.last_observation = obs
        act1 = self.policy_fc1.forward(obs)
        act2 = self.policy_fc2.forward(act1)
        logits = self.policy_output_fc.forward(act2)  # get action scores
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Method that calls value neural network to get a value estimate. Note that it does not take any parameters,
        so the last known observation needs to be saved on the model during the forward() call.
        Seems to be an RLLib thing.
        :return:
        """
        self.value_calls += 1
        assert self.last_observation is not None, "must call forward() first"
        act1 = self.value_fc1.forward(self.last_observation)
        act2 = self.value_fc2.forward(act1)
        value = self.value_output_fc.forward(act2)  # get value estimate
        return value.squeeze(1)
