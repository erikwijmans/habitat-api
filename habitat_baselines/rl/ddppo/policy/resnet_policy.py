#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.models import resnet
from habitat_baselines.models.resnet_encoders import ResNetEncoder
from habitat_baselines.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.models.running_mean_and_var import RunningMeanAndVar
from habitat_baselines.rl.ppo import Net, Policy


class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid="pointgoal_with_gps_compass",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
            ),
            action_space.n,
        )


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self._n_input_goal = (
            observation_space.spaces[self.goal_sensor_uuid].shape[0] + 1
        )
        self.tgt_embeding = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_tgt_encoding(self, observations):
        goal_observations = observations[self.goal_sensor_uuid]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )

        return self.tgt_embeding(goal_observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        tgt_encoding = self.get_tgt_encoding(observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x += [tgt_encoding, prev_actions]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
