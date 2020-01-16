#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
import tempfile
import time
from collections import defaultdict, deque
from typing import Dict, List

import hydra
import lmdb
import msgpack_numpy
import numpy as np
import omegaconf
import pydash
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.float
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


class LengthGroupedSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, batch_size):
        self.lengths = lengths
        self.sort_priority = list(range(len(self.lengths)))
        self.sorted_ordering = list(range(len(self.lengths)))

        self.batch_size = batch_size
        self.shuffling = list(range(len(self.lengths) // self.batch_size))

    def __iter__(self):
        random.shuffle(self.sort_priority)
        self.sorted_ordering.sort(
            key=lambda k: (int(self.lengths[k] / 1.1), self.sort_priority[k]),
            reverse=True,
        )
        random.shuffle(self.shuffling)
        for index in self.shuffling:
            for bid in range(self.batch_size):
                yield self.sorted_ordering[index * self.batch_size + bid]

    def __len__(self):
        return len(self.shuffling) * self.batch_size


class IWTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories_env_dir, length, use_iw):
        super().__init__()
        self.trajectories_env_dir = trajectories_env_dir
        self.lmdb_env = None
        self.length = length

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, 2.5])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

    def __getitem__(self, index):
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.trajectories_env_dir,
                map_size=1 << 40,
                write=False,
                lock=False,
            )

        with self.lmdb_env.begin(buffers=True) as txn:
            obs, prev_actions, oracle_actions = msgpack_numpy.unpackb(
                txn.get(str(index).encode()), raw=False
            )

        for k, v in obs.items():
            obs[k] = torch.from_numpy(v)

        prev_actions = torch.from_numpy(prev_actions)
        oracle_actions = torch.from_numpy(oracle_actions)

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __len__(self):
        return self.length


@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config.pretty()}")

        self.trajectory_lengths = []
        slurm_job_id = os.environ.get("SLURM_JOB_ID", 0)
        self.trajectories_env_dir = (
            f"/scratch/slurm_tmpdir/{slurm_job_id}/trajectories.lmdb"
        )

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
            "state_dict": self.actor_critic.state_dict(),
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

    def _update_dataset(self, data_it):
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
        batch = batch_obs(observations, device=self.device)

        episodes = []
        dones = []
        skips = []
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
            skips.append(False)

        beta = self.baselines_cfg.trainer.dagger.p ** data_it

        collected_eps = 0
        with tqdm.tqdm(
            total=self.baselines_cfg.trainer.dagger.update_size
        ) as pbar, lmdb.open(
            self.trajectories_env_dir, map_size=1 << 40
        ) as lmdb_env, lmdb_env.begin(
            write=True
        ) as txn, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            while (
                collected_eps < self.baselines_cfg.trainer.dagger.update_size
            ):
                for i in range(self.envs.num_envs):
                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs["oracle_action"]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(ep, use_bin_type=True),
                        )

                        self.trajectory_lengths.append(len(ep))
                        pbar.update()
                        collected_eps += 1

                    if dones[i]:
                        episodes[i] = []

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch["oracle_action"][i].item(),
                        )
                    )

                if beta < 1.0:
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
                    actions = torch.where(
                        torch.rand_like(actions, dtype=torch.float) < beta,
                        batch["oracle_action"].long(),
                        actions,
                    )
                else:
                    actions = batch["oracle_action"].long()

                skips = batch["oracle_action"].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)

                prev_actions.copy_(actions)

                outputs = self.envs.step([a[0].item() for a in actions])

                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                batch = batch_obs(observations, self.device)

        self.envs.close()
        self.envs = None

    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
    ):
        T, N = corrected_actions.size()
        self.optimizer.zero_grad()

        recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            N,
            self.baselines_cfg.model.hidden_size,
            device=self.device,
        )

        distribution = self.actor_critic.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        loss = ((weights * loss).sum(0) / weights.sum(0)).mean()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        os.makedirs(self.trajectories_env_dir, exist_ok=True)

        with lmdb.open(
            self.trajectories_env_dir, map_size=1 << 40
        ) as lmdb_env, lmdb_env.begin(write=True) as txn:
            txn.drop(lmdb_env.open_db())

        self.device = (
            torch.device("cuda", self.baselines_cfg.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.envs = construct_envs(
            self.config, get_env_class(self.baselines_cfg.env.name)
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
            for dagger_it in range(
                self.baselines_cfg.trainer.dagger.dagger_iters
            ):
                self._update_dataset(dagger_it)
                sampler = LengthGroupedSampler(
                    self.trajectory_lengths,
                    self.baselines_cfg.trainer.dagger.batch_size,
                )
                dataset = IWTrajectoryDataset(
                    self.trajectories_env_dir,
                    len(self.trajectory_lengths),
                    self.baselines_cfg.trainer.dagger.use_iw,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.baselines_cfg.trainer.dagger.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=8,
                )

                for epoch in tqdm.trange(
                    self.baselines_cfg.trainer.dagger.epochs
                ):
                    for batch in tqdm.tqdm(
                        diter, total=len(diter), leave=False
                    ):
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch
                        observations_batch = {
                            k: v.to(device=self.device, non_blocking=True)
                            for k, v in observations_batch.items()
                        }

                        loss = self._update_agent(
                            observations_batch,
                            prev_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            not_done_masks.to(
                                device=self.device, non_blocking=True
                            ),
                            corrected_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            weights_batch.to(
                                device=self.device, non_blocking=True
                            ),
                        )

                        writer.add_scalar("train_loss", loss, step_id)
                        step_id += 1

                    self.save_checkpoint(
                        f"ckpt.{dagger_it * self.baselines_cfg.trainer.dagger.epochs + epoch}.pth"
                    )

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
        self._setup_actor_critic_agent(config.habitat_baselines.trainer.dagger)

        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])

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
