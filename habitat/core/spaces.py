#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Sized

import gym
from gym import Space


class EmptySpace(Space):
    """
    A ``gym.Space`` that reflects arguments space for action that doesn't have
    arguments. Needed for consistency ang always samples `None` value.
    """

    def sample(self):
        return None

    def contains(self, x):
        return False


class ActionSpace(gym.spaces.Dict):
    """
    A dictionary of ``EmbodiedTask`` actions and their argument spaces.

    .. code:: py

        self.observation_space = spaces.ActionSpace(
            "move": spaces.Dict({
                "position": spaces.Discrete(2),
                "velocity": spaces.Discrete(3)
                },
            "move_forward": EmptySpace,
            )
        )
    """

    def __init__(self, spaces):
        if isinstance(spaces, dict):
            self.spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            self.spaces = OrderedDict(spaces)
        self.actions_select = gym.spaces.Discrete(len(self.spaces))

    @property
    def n(self):
        return len(self.spaces)

    def sample(self):
        action_index = self.actions_select.sample()
        return {
            "action": list(self.spaces.keys())[action_index],
            "action_args": list(self.spaces.values())[action_index].sample(),
        }

    def contains(self, x):
        if not isinstance(x, dict) and {"action", "action_args"} not in x:
            return False
        if not self.spaces[x["action"]].contains(x["action_args"]):
            return False
        return True

    def __repr__(self):
        return (
            "ActionSpace("
            + ", ".join([k + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )


class ListSpace(Space):
    """
    A ``gym.Space`` that describes a list of other Space. Used to describe
    list of tokens ids, vectors and etc.

    .. code:: py

        observation_space = ListSpace(spaces.Discrete(
            dataset.question_vocab.get_size()))
    """

    def __init__(self, space, min_seq_length=0, max_seq_length=1 << 15):
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.space = space
        self.length_select = gym.spaces.Discrete(
            max_seq_length - min_seq_length
        )

    def sample(self):
        seq_length = self.length_select.sample() + self.min_seq_length
        return [self.space.sample() for _ in range(seq_length)]

    def contains(self, x):
        if not isinstance(x, Sized):
            return False

        if self.min_seq_length <= len(x) <= self.max_seq_length:
            return False

        return all([self.space.contains(el) for el in x])

    def __repr__(self):
        return (
            f"ListSpace({self.space}, min_seq_length="
            f"{self.min_seq_length}, max_seq_length={self.max_seq_length})"
        )
