import gc
import json
import os
import random
import time
import warnings
from collections import defaultdict
from typing import Dict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import (
    construct_envs,
    construct_envs_auto_reset_false,
)
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, transform_obs
from habitat_baselines.models.vln_baseline_policy import VLNBaselinePolicy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    """

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
    def __init__(
        self,
        trajectories_env_dir,
        length,
        use_iw,
        inflection_weight_coef=1.0,
        lmbd_map_size=1e9,
    ):
        super().__init__()
        self.trajectories_env_dir = trajectories_env_dir
        self.lmdb_env = None
        self.length = length
        self.lmbd_map_size = lmbd_map_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

    def __getitem__(self, index):
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.trajectories_env_dir,
                map_size=int(self.lmbd_map_size),
                readonly=True,
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
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.trajectory_lengths = []
        self.trajectories_env_dir = (
            f"trajectories_dirs/{self.config.RUN_NAME}/trajectories.lmdb"
        )

    def _setup_actor_critic_agent(self, config: Config) -> None:
        r"""Sets up actor critic and agent.

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        # Add TORCH_GPU_ID to VLN config for a ResNet layer
        config.defrost()
        config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        config.freeze()

        self.actor_critic = VLNBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            vln_config=config,
        )
        self.actor_critic.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.config.DAGGER.LR
        )

    def save_checkpoint(self, file_name) -> None:
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
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
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
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        if self.envs is None:
            self.envs = construct_envs(
                self.config, get_env_class(self.config.ENV_NAME)
            )

        recurrent_hidden_states = torch.zeros(
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

        observations = self.envs.reset()
        observations = transform_obs(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        batch = batch_obs(observations, self.device)

        episodes = [[] for _ in range(self.envs.num_envs)]
        skips = [False for _ in range(self.envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(self.envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        if self.config.DAGGER.P == 0.0:
            # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
            beta = 0.0
        else:
            beta = self.config.DAGGER.P ** data_it

        collected_eps = 0
        with tqdm.tqdm(
            total=self.config.DAGGER.UPDATE_SIZE
        ) as pbar, lmdb.open(
            self.trajectories_env_dir,
            map_size=int(self.config.DAGGER.LMDB_MAP_SIZE),
        ) as lmdb_env, lmdb_env.begin(
            write=True
        ) as txn, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]

            while collected_eps < self.config.DAGGER.UPDATE_SIZE:
                for i in range(self.envs.num_envs):
                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs["vln_oracle_action_sensor"]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
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
                            batch["vln_oracle_action_sensor"][i].item(),
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
                        batch["vln_oracle_action_sensor"].long(),
                        actions,
                    )
                else:
                    actions = batch["vln_oracle_action_sensor"].long()

                skips = batch["vln_oracle_action_sensor"].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)

                prev_actions.copy_(actions)

                outputs = self.envs.step([a[0].item() for a in actions])
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                observations = transform_obs(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
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
            self.config.VLN.STATE_ENCODER.hidden_size,
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
        r"""Main method for training DAgger.

        Returns:
            None
        """
        os.makedirs(self.trajectories_env_dir, exist_ok=True)

        with lmdb.open(
            self.trajectories_env_dir,
            map_size=int(self.config.DAGGER.LMDB_MAP_SIZE),
        ) as lmdb_env, lmdb_env.begin(write=True) as txn:
            txn.drop(lmdb_env.open_db())

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        self._setup_actor_critic_agent(self.config.VLN)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )
        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    p.numel()
                    for p in self.actor_critic.parameters()
                    if p.requires_grad
                )
            )
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for dagger_it in range(self.config.DAGGER.ITERATIONS):
                step_id = 0
                self._update_dataset(dagger_it)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                gc.collect()

                sampler = LengthGroupedSampler(
                    self.trajectory_lengths, self.config.DAGGER.BATCH_SIZE
                )
                dataset = IWTrajectoryDataset(
                    self.trajectories_env_dir,
                    len(self.trajectory_lengths),
                    self.config.DAGGER.USE_IW,
                    inflection_weight_coef=self.config.VLN.inflection_weight_coef,
                    lmbd_map_size=self.config.DAGGER.LMDB_MAP_SIZE,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    shuffle=False,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=8,
                )

                for epoch in tqdm.trange(self.config.DAGGER.EPOCHS):
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

                        try:
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
                        except:
                            logger.info(
                                "ERROR: failed to update agent. Updating agent with batch size of 1."
                            )
                            loss = 0
                            prev_actions_batch = prev_actions_batch.cpu()
                            not_done_masks = not_done_masks.cpu()
                            corrected_actions_batch = (
                                corrected_actions_batch.cpu()
                            )
                            weights_batch = weights_batch.cpu()
                            observations_batch = {
                                k: v.cpu()
                                for k, v in observations_batch.items()
                            }
                            for i in range(not_done_masks.size(0)):
                                loss += self._update_agent(
                                    {
                                        k: v[i].to(
                                            device=self.device,
                                            non_blocking=True,
                                        )
                                        for k, v in observations_batch.items()
                                    },
                                    prev_actions_batch[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                    not_done_masks[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                    corrected_actions_batch[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                    weights_batch[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                )

                        logger.info(f"train_loss: {loss}")
                        logger.info(f"Batches processed: {step_id}.")
                        logger.info(
                            f"On DAgger iter {dagger_it}, Epoch {epoch}."
                        )
                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        step_id += 1

                    self.save_checkpoint(
                        f"ckpt.{dagger_it * self.config.DAGGER.EPOCHS + epoch}.pth"
                    )

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Assumes episode IDs are unique.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        # setup agent
        self.envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._setup_actor_critic_agent(config.VLN)
        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])

        observations = self.envs.reset()
        observations = transform_obs(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        batch = batch_obs(observations, self.device)

        eval_recurrent_hidden_states = torch.zeros(
            1,  # num_recurrent_layers
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

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            rgb_frames = [[] for _ in range(self.config.NUM_PROCESSES)]

        self.actor_critic.eval()
        while (
            self.envs.num_envs > 0
            and len(stats_episodes) < self.config.TEST_EPISODE_COUNT
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    eval_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    eval_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                stats_episodes[current_episodes[i].episode_id] = infos[i]
                observations[i] = self.envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)

                if len(self.config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=self.config.VIDEO_OPTION,
                        video_dir=self.config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metric_name="SPL",
                        metric_value=round(
                            stats_episodes[current_episodes[i].episode_id][
                                "spl"
                            ],
                            6,
                        ),
                        tb_writer=writer,
                    )

                    del stats_episodes[current_episodes[i].episode_id][
                        "top_down_map"
                    ]
                    del stats_episodes[current_episodes[i].episode_id][
                        "collisions"
                    ]
                    rgb_frames[i] = []

            observations = transform_obs(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)

            envs_to_pause = []
            next_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
            )

        self.envs.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        split = config.TASK_CONFIG.DATASET.SPLIT
        with open(f"stats_episodes_{checkpoint_index}_{split}.json", "w") as f:
            json.dump(aggregated_stats, f, indent=4)

        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalar(f"eval_{k}", v, checkpoint_num)
