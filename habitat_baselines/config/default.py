#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import inspect
import os
import os.path as osp
from typing import List, Optional, Union

from habitat.config import get_config
from habitat.config.default import HabitatHydraLoader

HabitatHydraLoader().provider_names.append("habitat_baselines")
HabitatHydraLoader().provider_paths.append(
    "pkg://habitat_baselines.config.base"
)
HabitatHydraLoader().base_configs.append("habitat_baselines_base.yaml")
