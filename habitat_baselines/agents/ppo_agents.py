#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random

import numpy as np
import omegaconf
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat_baselines.common.utils import batch_obs
from habitat_baselines.rl.ppo import PointNavBaselinePolicy


def get_default_config():
    cfg = omegaconf.OmegaConf.create(
        dict(
            input_type="blind",
            model_path="data/checkpoints/blind.pth",
            resolution=256,
            hidden_size=512,
            random_seed=7,
            pth_gpu_id=0,
            goal_sensor_uuid="pointgoal_with_gps_compass",
        )
    )

    omegaconf.OmegaConf.set_struct(cfg, True)

    return cfg


class PPOAgent(Agent):
    def __init__(self, config: Config):
        self.goal_sensor_uuid = config.goal_sensor_uuid
        spaces = {
            self.goal_sensor_uuid: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if config.input_type in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(config.resolution, config.resolution, 1),
                dtype=np.float32,
            )

        if config.input_type in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(config.resolution, config.resolution, 3),
                dtype=np.uint8,
            )
        observation_spaces = Dict(spaces)

        action_spaces = Discrete(4)

        self.device = (
            torch.device("cuda:{}".format(config.pth_gpu_id))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.hidden_size

        random.seed(config.random_seed)
        torch.random.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

        self.actor_critic = PointNavBaselinePolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            goal_sensor_uuid=self.goal_sensor_uuid,
        )
        self.actor_critic.to(self.device)

        if config.model_path:
            ckpt = torch.load(config.model_path, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations):
        batch = batch_obs([observations])
        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks = torch.ones(1, 1, device=self.device)
            self.prev_actions.copy_(actions)

        return {"action": actions[0][0].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    args = parser.parse_args()

    config = get_config(args.task_config)

    agent_config = get_default_config()
    agent_config.input_type = args.input_type
    agent_config.model_path = args.model_path
    agent_config.goal_sensor_uuid = config.task.goal_sensor_uuid

    agent = PPOAgent(agent_config)
    benchmark = habitat.Benchmark(config_paths=args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
