#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import List, Optional, Union

import numpy as np

from habitat.config import get_config
from habitat.config.default import HabitatBaseConfigSearcher

HabitatBaseConfigSearcher.add_recursive_search_path(
    osp.join(osp.abspath(osp.dirname(__file__)), "base")
)
