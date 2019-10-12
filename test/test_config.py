#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.config.default import get_config

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"
CFG_EQA = "configs/test/habitat_mp3d_eqa_test.yaml"
MAX_TEST_STEPS_LIMIT = 3


def test_merged_configs():
    test_config = get_config(CFG_TEST)
    eqa_config = get_config(CFG_EQA)
    merged_config = get_config("{},{}".format(CFG_TEST, CFG_EQA))
    assert merged_config.task.type == eqa_config.task.type
    assert (
        merged_config.environment.max_episode_steps
        == test_config.environment.max_episode_steps
    )


def test_overwrite_options():
    for steps_limit in range(MAX_TEST_STEPS_LIMIT):
        config = get_config(
            config_paths=CFG_TEST,
            opts=["environment.max_episode_steps", steps_limit],
        )
        assert (
            config.environment.max_episode_steps == steps_limit
        ), "Overwriting of config options failed."
