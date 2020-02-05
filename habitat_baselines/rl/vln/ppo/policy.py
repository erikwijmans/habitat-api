#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from gym import Space

from habitat import Config
from habitat_baselines.models.vln_baseline_policy import VLNBaselineNet
from habitat_baselines.rl.ppo.policy import CriticHeadMLP, Policy


class VLNBaselineRLPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, vln_config: Config
    ):
        super().__init__(
            VLNBaselineNet(
                observation_space=observation_space, vln_config=vln_config
            ),
            action_space.n,
        )
        # Use a non-linear critic regressor
        self.critic = CriticHeadMLP(self.net.output_size)
