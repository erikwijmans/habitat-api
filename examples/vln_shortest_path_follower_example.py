#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import shutil

import cv2
import numpy as np

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def save_map(observations, info, images):
    im = observations["rgb"]
    top_down_map = draw_top_down_map(
        info, observations["heading"], im.shape[0]
    )
    output_im = np.concatenate((im, top_down_map), axis=1)
    shape = output_im.shape
    color = (255, 0, 0)
    org = (5, shape[0] - 10)

    fontScale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    y0, dy = shape[0] - 80, 20
    for i, line in enumerate(observations["instruction"]["text"].split(".")):
        y = y0 + i * dy
        cv2.putText(
            output_im,
            line,
            (5, y),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    images.append(output_im)


def shortest_path_example(mode, all_episodes=False):
    """
    Saves a video of a shortest path follower agent navigating from a start
    position to a goal. Agent navigates to intermediate viewpoints on the way.
    Args:
        mode: 'geodesic_path' or 'greedy'
        all_episodes: if True, runs for every episode. otherwise, 5.
    """
    config = habitat.get_config(
        config_paths="configs/test/habitat_r2r_vln_test.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    env = SimpleRLEnv(config=config)

    follower = ShortestPathFollower(
        env.habitat_env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = mode
    print("Environment creation successful")

    dirname = os.path.join(IMAGE_DIR, "vln_path_follow")
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
        os.makedirs(dirname)

    episodes_range = len(env.episodes) if all_episodes else 1
    for episode in range(episodes_range):
        env.reset()
        episode_id = env.habitat_env.current_episode.episode_id
        print(
            f"Agent stepping around inside environment. Episode id: {episode_id}"
        )

        images = []
        steps = 0
        path = env.habitat_env.current_episode.path + [
            env.habitat_env.current_episode.goals[0].position
        ]
        for point in path:
            done = False
            while not done:
                best_action = follower.get_next_action(point)
                if best_action == None:
                    break
                observations, reward, done, info = env.step(best_action)
                save_map(observations, info, images)
                steps += 1

        print(f"Navigated to goal in {steps} steps.")
        images_to_video(images, dirname, str(episode_id))
        images = []


if __name__ == "__main__":
    shortest_path_example("geodesic_path")
