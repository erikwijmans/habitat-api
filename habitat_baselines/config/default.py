#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

from habitat.config import get_config as get_hab_config
from habitat.config.default import HabitatHydraLoader
from habitat.core.utils import Singleton


class HabitatBaselinesConfigLoader(metaclass=Singleton):
    def __init__(self):
        HabitatHydraLoader().provider_names.append("habitat_baselines")
        HabitatHydraLoader().provider_paths.append(
            "pkg://habitat_baselines.config.base"
        )
        HabitatHydraLoader().base_configs.append("habitat_baselines_base.yaml")


@functools.wraps(get_hab_config)
def get_config(config_paths, overrides):
    HabitatBaselinesConfigLoader()

    return get_hab_config(config_paths, overrides)
