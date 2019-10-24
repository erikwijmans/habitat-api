#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import omegaconf
import pytest

from habitat.config.default import get_config
from habitat.sims import make_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def init_sim():
    config = get_config()
    if not os.path.exists(config.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")
    return make_sim(config.simulator.type, config=config.simulator)


def test_sim_trajectory():
    with open("test/data/habitat-sim_trajectory_data.json", "r") as f:
        test_trajectory = json.load(f)
    sim = init_sim()

    sim.reset()
    sim.set_agent_state(
        position=test_trajectory["positions"][0],
        rotation=test_trajectory["rotations"][0],
    )

    # remove last stop action as Sim has no stop action anymore
    for i, action in enumerate(test_trajectory["actions"][:-1]):
        action = HabitatSimActions[action.lower()]
        if i > 0:  # ignore first step as habitat-sim doesn't update
            # agent until then
            state = sim.get_agent_state()
            assert (
                np.allclose(
                    np.array(
                        test_trajectory["positions"][i], dtype=np.float32
                    ),
                    state.position,
                )
                is True
            ), "mismatch in position " "at step {}".format(i)
            assert (
                np.allclose(
                    np.array(
                        test_trajectory["rotations"][i], dtype=np.float32
                    ),
                    np.array([*state.rotation.imag, state.rotation.real]),
                )
                is True
            ), "mismatch in rotation " "at step {}".format(i)

            max_search_radius = 2.0
            dist_to_obs = sim.distance_to_closest_obstacle(
                state.position, max_search_radius
            )
            assert np.isclose(
                dist_to_obs, test_trajectory["distances_to_obstacles"][i]
            )

        assert sim.action_space.contains(action)

        sim.step(action)

    sim.close()


def test_sim_no_sensors():
    config = get_config()

    with omegaconf.read_write(config):
        config.simulator.agent_0.sensors = []

    if not os.path.exists(config.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")
    sim = make_sim(config.simulator.type, config=config.simulator)
    sim.reset()
    sim.close()
