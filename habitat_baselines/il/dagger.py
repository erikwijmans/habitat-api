#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
import time
from collections import defaultdict, deque
from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import pydash
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer, BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PointNavBaselinePolicy


@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config.pretty()}")

        self.dataset = []

    def _setup_actor_critic_agent(self, dagger_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.baselines_cfg.logging.file)

        self.actor_critic = PointNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=self.baselines_cfg.model.hidden_size,
            goal_sensor_uuid=self.config.habitat.task.goal_sensor_uuid,
        )
        self.actor_critic.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=dagger_cfg.lr
        )

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint,
            os.path.join(
                self.baselines_cfg.logging.checkpoint_folder, file_name
            ),
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_dataset(self, epoch):
        torch.cuda.empty_cache()
        if self.envs is None:
            self.envs = construct_envs(
                self.config, get_env_class(self.baselines_cfg.env.name)
            )
        recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.baselines_cfg.num_processes,
            self.baselines_cfg.model.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.baselines_cfg.num_processes,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            self.baselines_cfg.num_processes, 1, device=self.device
        )

        observations = self.envs.reset()
        batch = batch_obs(observations)

        episodes = []
        dones = []
        for i in range(self.envs.num_envs):
            episodes.append(
                [
                    (
                        observations[i],
                        prev_actions[i].item(),
                        batch["oracle_action"][i].item(),
                    )
                ]
            )
            dones.append(False)

        curr_dataset_size = len(self.dataset)
        beta = 0.5 ** (epoch - 1)
        while (
            len(self.dataset) - curr_dataset_size
        ) < self.baselines_cfg.trainer.dagger.update_size:
            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

            for i in range(self.envs.num_envs):
                episodes[i].append(
                    (
                        observations[i],
                        prev_actions[i].item(),
                        batch["oracle_action"][i].item(),
                    )
                )

                if dones[i]:
                    ep = episodes[i]
                    traj_obs = batch_obs(
                        [step[0] for step in ep], device=torch.device("cpu")
                    )
                    del traj_obs["oracle_action"]

                    self.dataset.append(
                        (
                            traj_obs,
                            torch.tensor(
                                [step[1] for step in ep], dtype=torch.long
                            ),
                            torch.tensor(
                                [step[2] for step in ep], dtype=torch.long
                            ),
                        )
                    )

                    episodes[i] = []

            actions = torch.where(
                torch.rand_like(actions, dtype=torch.float) < beta,
                actions,
                batch["oracle_action"].long(),
            )
            prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            batch = batch_obs(observations, self.device)

        self.envs.close()
        self.envs = None

        self.dataset.sort(key=lambda v: v[1].size(0))

    def _update_agent(
        self, observations, prev_actions, corrected_actions, not_done_masks
    ):
        self.optimizer.zero_grad()

        recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.baselines_cfg.trainer.dagger.batch_size,
            self.baselines_cfg.model.hidden_size,
            device=self.device,
        )

        distribution = self.actor_critic.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        mask = corrected_actions != -1

        corrected_actions = corrected_actions[mask]
        logits = logits[mask.view(-1, 1).expand_as(logits)].view(
            -1, logits.size(1)
        )
        loss = F.cross_entropy(logits, corrected_actions)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.baselines_cfg.env.name)
        )

        self.device = (
            torch.device("cuda", self.baselines_cfg.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        os.makedirs(
            self.baselines_cfg.logging.checkpoint_folder, exist_ok=True
        )
        self._setup_actor_critic_agent(self.baselines_cfg.trainer.dagger)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )

        with TensorboardWriter(
            self.baselines_cfg.tensorboard.dir,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            step_id = 0
            for epoch in range(self.baselines_cfg.trainer.dagger.epochs):
                self._update_dataset(epoch)
                B = self.baselines_cfg.trainer.dagger.batch_size
                num_steps = len(self.dataset) // B
                ordering = list(range(num_steps))
                random.shuffle(ordering)
                for i in range(num_steps):
                    observations_batch = defaultdict(list)
                    prev_actions_batch = []
                    corrected_actions_batch = []

                    idx = ordering[i]

                    # Select the trajectories in a batch.
                    # Trajectories are selected seqentially to keep things sorted by
                    # length to minimize
                    for bid in range(B):
                        traj = self.dataset[idx * B + bid]
                        for sensor in traj[0]:
                            observations_batch[sensor].append(traj[0][sensor])

                        prev_actions_batch.append(traj[1])
                        corrected_actions_batch.append(traj[2])

                    max_traj_len = max(
                        ele.size(0) for ele in prev_actions_batch
                    )
                    for bid in range(B):
                        for sensor in observations_batch:
                            curr = observations_batch[sensor][bid]
                            observations_batch[sensor][bid] = torch.cat(
                                [curr]
                                + [
                                    curr[0:1]
                                    for _ in range(max_traj_len - curr.size(0))
                                ],
                                dim=0,
                            )

                        curr = prev_actions_batch[bid]
                        prev_actions_batch[bid] = torch.cat(
                            [curr]
                            + [
                                curr[0:1]
                                for _ in range(max_traj_len - curr.size(0))
                            ],
                            dim=0,
                        )

                        curr = corrected_actions_batch[bid]
                        corrected_actions_batch[bid] = torch.cat(
                            [curr]
                            + [
                                curr[0:1].clone().fill_(-1)
                                for _ in range(max_traj_len - curr.size(0))
                            ],
                            dim=0,
                        )

                    for sensor in observations_batch:
                        observations_batch[sensor] = torch.stack(
                            observations_batch[sensor], dim=1
                        )
                        observations_batch[sensor] = (
                            observations_batch[sensor]
                            .view(-1, *observations_batch[sensor].size()[2:])
                            .to(device=self.device)
                        )

                    prev_actions_batch = torch.stack(
                        prev_actions_batch, dim=1
                    ).to(device=self.device)
                    corrected_actions_batch = torch.stack(
                        corrected_actions_batch, dim=1
                    ).to(device=self.device)
                    not_done_masks = torch.ones_like(
                        corrected_actions_batch, dtype=torch.float
                    )
                    not_done_masks[0] = 0

                    loss = self._update_agent(
                        observations_batch,
                        prev_actions_batch.view(-1, 1),
                        corrected_actions_batch.view(-1),
                        not_done_masks.view(-1, 1),
                    )

                    writer.add_scalars("loss", {"train": loss}, step_id)
                    step_id += 1

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.baselines_cfg.eval.use_ckpt_config:
            config = ckpt_dict["config"]
        else:
            config = copy.deepcopy(self.config)

        # get name of performance metric, e.g. "spl"
        self.metric_uuids = []
        for metric_name in config.habitat.task.measure:
            metric_cfg = getattr(config.habitat.task.measure, metric_name)
            measure_init = baseline_registry.get_measure(metric_cfg.type)
            assert (
                measure_init is not None
            ), "invalid measurement type {}".format(metric_cfg.type)
            self.metric_uuids.append(
                measure_init(sim=None, task=None, config=None)._get_uuid()
            )

        ppo_cfg = config.habitat_baselines.trainer.ppo

        with omegaconf.read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.baselines_cfg.video.outputs) > 0:
            extra_measures_cfg = hydra.experimental.compose(
                overrides=[
                    "habitat/task/measure=top_down_map",
                    "habitat/task/measure=collisions",
                ]
            )
            config = omegaconf.OmegaConf.merge(config, extra_measures_cfg)

        logger.info(f"env config: {config}")
        self.envs = construct_envs(
            config, get_env_class(config.habitat_baselines.env.name)
        )
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.baselines_cfg.num_processes,
            self.baselines_cfg.model.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.baselines_cfg.num_processes,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            self.baselines_cfg.num_processes, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.baselines_cfg.num_processes)
        ]  # type: List[List[np.ndarray]]
        if len(self.baselines_cfg.video.outputs) > 0:
            os.makedirs(self.baselines_cfg.video.dir, exist_ok=True)

        while (
            len(stats_episodes) < self.baselines_cfg.eval.test_episode_count
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]

                    episode_stats["reward"] = current_episode_reward[i].item()
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.baselines_cfg.video.outputs) > 0:
                        generate_video(
                            video_option=self.baselines_cfg.video.outputs,
                            video_dir=self.baselines_cfg.video.dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name=self.metric_uuid,
                            metric_value=infos[i][self.metric_uuid],
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.baselines_cfg.video.outputs) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        writer.add_scalars(
            "eval_reward",
            {"average reward": episode_reward_mean},
            checkpoint_index,
        )

        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: {aggregated_stats[metric_uuid]/num_episodes:.6f}"
            )
            writer.add_scalars(
                f"eval_{metric_uuid}",
                {
                    f"average {metric_uuid}": aggregated_stats[metric_uuid]
                    / num_episodes
                },
                checkpoint_index,
            )

        self.envs.close()
