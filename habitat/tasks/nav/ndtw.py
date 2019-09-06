from typing import Any, List, Optional, Type

import numba
import numpy as np

from habitat.config import Config
from habitat.core.embodied_task import Episode, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, Simulator


def _discretize_path(path, sampling_rate):
    current_pt = path[0].copy()
    discrete_path = [current_pt.copy()]
    current_dir = path[1] - path[0]
    current_dir /= np.linalg.norm(current_dir)
    i = 1
    while i < len(path):
        current_pt = current_pt + sampling_rate * current_dir
        if np.linalg.norm(current_pt - path[i - 1]) > np.linalg.norm(
            path[i] - path[i - 1]
        ):
            current_pt = path[i].copy()
            i += 1

            if i < len(path):
                current_dir = path[i] - path[i - 1]
                current_dir /= np.linalg.norm(current_dir)

        discrete_path.append(current_pt)

    return discrete_path


@registry.register_measure
class NDTW(Measure):
    r"""Normalized Dynamic Time Warping
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config

        self._sampling_rate = 0.025
        self._dth = 0.25
        # The orginal paper sets this to 1, but that appears to be much to strict
        # when you are comparing paths in habitat-sim
        self._max_warp_dist = 5
        self._q = []
        self._r = []

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ndtw"

    def reset_metric(self, episode: Episode):
        start_pos = np.array(episode.start_position)
        ref_path = self._sim.get_straight_shortest_path_points(
            start_pos, episode.goals[0].position
        )
        self._r = _discretize_path(ref_path, self._sampling_rate)
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._q = [start_pos]

        self._metric = None

    def update_metric(self, episode, action):
        new_pos = self._sim.get_agent_state().position
        if np.linalg.norm(new_pos - self._q[-1]) >= 1e-2:
            self._q.append(new_pos)

        if action == self._sim.index_stop_action:
            dtw = self._sim._sim.pathfinder.dtw(
                self._r,
                _discretize_path(self._q, self._sampling_rate),
                self._max_warp_dist,
            )
            self._metric = np.exp(-(dtw / (len(self._r) * self._dth)))
        else:
            self._metric = 0.0
