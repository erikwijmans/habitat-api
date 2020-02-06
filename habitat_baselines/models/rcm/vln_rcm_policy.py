#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from habitat import Config
from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.models.instruction_encoder import InstructionEncoder
from habitat_baselines.models.rcm.rcm_state_encoder import RCMStateEncoder
from habitat_baselines.models.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from habitat_baselines.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.models.simple_cnn import (
    SimpleCNN,
    SimpleDepthCNN,
    SimpleRGBCNN,
)
from habitat_baselines.rl.ppo.policy import Net, Policy


class VLNRCMPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, vln_config: Config
    ):
        super().__init__(
            VLNRCMNet(
                observation_space=observation_space,
                vln_config=vln_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class VLNRCMNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.

    Modules:
        Instruction encoder
        Depth encoder
        Visual (RGB) encoder
        RNN state decoder or RCM state_encoder
    """

    def __init__(
        self, observation_space: Space, vln_config: Config, num_actions
    ):
        super().__init__()
        self.vln_config = vln_config
        vln_config.defrost()
        vln_config.INSTRUCTION_ENCODER.final_state_only = False
        vln_config.freeze()

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            vln_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert vln_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=vln_config.DEPTH_ENCODER.output_size,
            checkpoint=vln_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=vln_config.DEPTH_ENCODER.backbone,
            spatial_output=True,
        )

        # Init the RGB visual encoder
        assert vln_config.VISUAL_ENCODER.cnn_type in [
            "TorchVisionResNet50"
        ], "VISUAL_ENCODER.cnn_type must be TorchVisionResNet50'."

        device = (
            torch.device("cuda", vln_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.visual_encoder = TorchVisionResNet50(
            observation_space,
            vln_config.VISUAL_ENCODER.output_size,
            device,
            activation=vln_config.VISUAL_ENCODER.activation,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        self.rcm_state_encoder = vln_config.RCM.rcm_state_encoder

        hidden_size = vln_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size
        if self.rcm_state_encoder:
            self.state_encoder = RCMStateEncoder(
                self.visual_encoder.output_shape[0],
                self.depth_encoder.output_shape[0],
                vln_config.STATE_ENCODER.hidden_size,
                self.prev_action_embedding.embedding_dim,
            )
        else:
            self.rgb_linear = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(
                    self.visual_encoder.output_shape[0],
                    self.vln_config.VISUAL_ENCODER.output_size,
                ),
                nn.ReLU(True),
            )
            self.depth_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.depth_encoder.output_shape),
                    self.vln_config.DEPTH_ENCODER.output_size,
                ),
                nn.ReLU(True),
            )

            # Init the RNN state decoder
            rnn_input_size = vln_config.DEPTH_ENCODER.output_size
            rnn_input_size += vln_config.VISUAL_ENCODER.output_size
            rnn_input_size += self.prev_action_embedding.embedding_dim

            self.state_encoder = RNNStateEncoder(
                input_size=rnn_input_size,
                hidden_size=vln_config.STATE_ENCODER.hidden_size,
                num_layers=1,
                rnn_type=vln_config.STATE_ENCODER.rnn_type,
            )

        self.rgb_kv = nn.Conv1d(
            self.visual_encoder.output_shape[0],
            hidden_size // 2 + vln_config.VISUAL_ENCODER.output_size,
            1,
        )
        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + vln_config.DEPTH_ENCODER.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.train()

    @property
    def output_size(self):
        return (
            self.vln_config.STATE_ENCODER.hidden_size
            + self.vln_config.VISUAL_ENCODER.output_size
            + self.vln_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
        )

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x VISUAL_ENCODER.output_size]
        """
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.visual_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.rcm_state_encoder:
            state, rnn_hidden_states = self.state_encoder(
                rgb_embedding,
                depth_embedding,
                prev_actions,
                rnn_hidden_states,
                masks,
            )
        else:
            rgb_in = self.rgb_linear(rgb_embedding)
            depth_in = self.depth_linear(depth_embedding)

            state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
            state, rnn_hidden_states = self.state_encoder(
                state_in, rnn_hidden_states, masks
            )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)

        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [state, text_embedding, rgb_embedding, depth_embedding], dim=1
        )

        return x, rnn_hidden_states


if __name__ == "__main__":
    from habitat_baselines.config.default import get_config
    from gym import spaces

    config = get_config("habitat_baselines/config/vln/il_vln.yaml")

    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    observation_space = spaces.Dict(
        dict(
            rgb=spaces.Box(
                low=0, high=0, shape=(224, 224, 3), dtype=np.float32
            ),
            depth=spaces.Box(
                low=0, high=0, shape=(256, 256, 1), dtype=np.float32
            ),
        )
    )

    # Add TORCH_GPU_ID to VLN config for a ResNet layer
    config.defrost()
    config.VLN.TORCH_GPU_ID = config.TORCH_GPU_ID
    config.freeze()

    action_space = spaces.Discrete(4)

    policy = VLNRCMPolicy(observation_space, action_space, config.VLN).to(
        device
    )

    dummy_instruction = torch.randint(1, 4, size=(4 * 2, 8), device=device)
    dummy_instruction[:, 5:] = 0
    dummy_instruction[0, 2:] = 0

    obs = dict(
        rgb=torch.randn(4 * 2, 224, 224, 3, device=device),
        depth=torch.randn(4 * 2, 256, 256, 1, device=device),
        instruction=dummy_instruction,
    )

    hidden_states = torch.randn(
        policy.net.state_encoder.num_recurrent_layers,
        2,
        policy.net._hidden_size,
        device=device,
    )
    prev_actions = torch.randint(0, 3, size=(4 * 2, 1), device=device)
    masks = torch.ones(4 * 2, 1, device=device)

    policy.evaluate_actions(
        obs, hidden_states, prev_actions, masks, prev_actions
    )
