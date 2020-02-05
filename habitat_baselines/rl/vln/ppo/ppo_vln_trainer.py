#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    observations_to_image,
)
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    transform_obs,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.vln.ppo import VLNBaselineRLPolicy


@baseline_registry.register_trainer(name="ppo_vln")
class PPOVLNTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_actor_critic_agent_training(self, cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(cfg.LOG_FILE)

        # Add TORCH_GPU_ID to VLN config for a ResNet layer
        cfg.defrost()
        cfg.VLN.TORCH_GPU_ID = cfg.TORCH_GPU_ID
        cfg.freeze()

        self.actor_critic = VLNBaselineRLPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            vln_config=cfg.VLN,
        )
        self.actor_critic.to(self.device)

        if cfg.RL.LOAD_FROM_CKPT:
            # assumes config params are the same. Manually removes first
            # keyword to conform to expected key name.
            logger.info(f"Loading checkpoint: {cfg.RL.CKPT_TO_LOAD}")
            il_checkpoint = self.load_checkpoint(
                cfg.RL.CKPT_TO_LOAD, map_location=self.device
            )
            logger.info(
                f"Loading weights expecting CONFIG.RL.VLN params: {il_checkpoint['config']['RL']['VLN']}"
            )

            il_checkpoint_dict = {}
            for k, v in il_checkpoint["state_dict"].items():
                layer_name = ".".join(k.split(".")[1:])
                il_checkpoint_dict[layer_name] = v

            del il_checkpoint
            not_updated = [
                k
                for k in self.actor_critic.state_dict().keys()
                if k not in il_checkpoint_dict.keys()
            ]
            if not_updated:
                logger.warn(
                    f"Layers not loaded from checkpoint ({len(not_updated)}):\n"
                    + "\n".join(not_updated)
                )

            self.actor_critic.load_state_dict(il_checkpoint_dict, strict=False)
            del il_checkpoint_dict

        ppo_cfg = cfg.RL.PPO
        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_clipped_value_loss=ppo_cfg.use_clipped_value_loss,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            use_different_critic_lr=ppo_cfg.use_different_critic_lr,
            critic_lr=ppo_cfg.critic_lr,
        )

    def _setup_actor_critic_agent(self, cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            rl_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(cfg.LOG_FILE)

        # Add TORCH_GPU_ID to VLN config for a ResNet layer
        cfg.defrost()
        cfg.VLN.TORCH_GPU_ID = cfg.TORCH_GPU_ID
        cfg.freeze()

        self.actor_critic = VLNBaselineRLPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            vln_config=cfg.VLN,
        )
        self.actor_critic.to(self.device)

        ppo_cfg = cfg.RL.PPO
        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_clipped_value_loss=ppo_cfg.use_clipped_value_loss,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            use_different_critic_lr=ppo_cfg.use_different_critic_lr,
            critic_lr=ppo_cfg.critic_lr,
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
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
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

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, episode_rewards, episode_counts
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        observations = transform_obs(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        batch = batch_obs(observations, self.device)

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_counts += 1 - masks
        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs, infos, dones

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        rl_config = self.config.RL
        ppo_cfg = rl_config.PPO

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent_training(self.config)
        logger.info(
            "agent number of parameters: {}".format(
                sum(p.numel() for p in self.agent.parameters())
            )
        )
        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    p.numel()
                    for p in self.agent.parameters()
                    if p.requires_grad
                )
            )
        )

        # set the observation space for instructions according to config
        for i in range(len(self.envs.observation_spaces)):
            self.envs.observation_spaces[i].spaces["instruction"] = spaces.Box(
                low=0,
                high=self.config.VLN.INSTRUCTION_ENCODER.vocab_size,
                shape=(self.config.VLN.INSTRUCTION_ENCODER.max_length,),
                dtype=np.float32,
            )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            self.config.VLN.STATE_ENCODER.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        observations = transform_obs(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        batch = batch_obs(observations, self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_metrics = {}
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert (
                measure_type is not None
            ), "invalid measurement type {}".format(metric_cfg.TYPE)
            window_metrics[measure_type._get_uuid()] = deque(
                maxlen=ppo_cfg.reward_window_size
            )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                infos_list = []
                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        infos,
                        dones,
                    ) = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        episode_rewards,
                        episode_counts,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                    for done, info in zip(dones, infos):
                        if done:
                            infos_list.append(info)

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)

                # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                for info in infos_list:
                    for k in window_metrics.keys():
                        window_metrics[k].append(info[k])

                losses = [value_loss, action_loss]
                stats = zip(
                    ["count", "reward"],
                    [window_episode_counts, window_episode_reward],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                for k, v in window_metrics.items():
                    writer.add_scalar(k, np.mean(v), count_steps)

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                writer.add_scalar("entropy", dist_entropy, count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                        for k, v in window_metrics.items():
                            logger.info(
                                "Average window size {} {}: {:3f}".format(
                                    len(v), k, np.mean(v)
                                )
                            )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

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

        if self.config.EVAL.USE_CKPT_CONFIG:
            self.config = self._setup_eval_config(ckpt_dict["config"])
        else:
            self.config = self.config.clone()

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            self.config.defrost()
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            self.config.freeze()

        logger.info(f"env config: {self.config}")
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        self._setup_actor_critic_agent(self.config)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        observations = transform_obs(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.VLN.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
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
            observations = transform_obs(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
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
                    video_metric_uuid = None
                    for (
                        metric_name
                    ) in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
                        metric_cfg = getattr(
                            self.config.TASK_CONFIG.TASK, metric_name
                        )
                        measure_type = baseline_registry.get_measure(
                            metric_cfg.TYPE
                        )
                        assert (
                            measure_type is not None
                        ), "invalid measurement type {}".format(
                            metric_cfg.TYPE
                        )
                        metric_uuid = measure_type._get_uuid()
                        if not video_metric_uuid:
                            video_metric_uuid = metric_uuid

                        # only support scalar metrics
                        if type(infos[i][metric_uuid]) != dict:
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

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name=video_metric_uuid,
                            metric_value=infos[i][video_metric_uuid],
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
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

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalars(
                f"eval_{k}", {f"Average_{k}": v}, checkpoint_index
            )

        self.envs.close()
