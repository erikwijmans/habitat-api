#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from typing import Type, Union

import omegaconf

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset


def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]], rank: int
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.habitat.dataset.type, config=config.habitat.dataset
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(rank)
    return env


def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.

    Returns:
        VectorEnv object created according to specification.
    """

    num_processes = config.habitat_baselines.num_processes
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.habitat.dataset.type)
    scenes = dataset.get_scenes_to_load(config.habitat.dataset)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = copy.deepcopy(config)

        with omegaconf.read_write(proc_config):
            if len(scenes) > 0:
                proc_config.habitat.dataset.content_scenes = scene_splits[i]

            proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                config.habitat_baselines.simulator_gpu_id
            )

        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(configs, env_classes, range(num_processes)))
        ),
    )
    return envs
