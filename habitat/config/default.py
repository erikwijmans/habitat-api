#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import os.path as osp
from typing import List, Optional, Union

import hydra
import hydra.experimental
import omegaconf
from hydra._internal.hydra import GlobalHydra


def get_config(
    config_file: Optional[str] = None, overrides: Optional[List[str]] = None
):
    if not GlobalHydra().is_initialized():
        hydra.experimental.initialize(caller_stack_depth=2)

    if not any(
        path.provider == "habitat"
        for path in GlobalHydra().hydra.config_loader.config_search_path.config_search_path
    ):
        GlobalHydra().hydra.config_loader.config_search_path.prepend(
            "habitat", "pkg://habitat.config.base"
        )

    action_order = []
    cfg = hydra.experimental.compose("habitat_base.yaml", [])
    if config_file is not None:
        extended_cfg_defaults = hydra.experimental.compose(config_file, [])
        extended_cfg_overrides = omegaconf.OmegaConf.load(config_file)
        if "defaults" in extended_cfg_overrides:
            defaults = extended_cfg_overrides.defaults

            for default in defaults:
                k, v = next(default.items())
                if "habitat/task/action" == k:
                    action_order.append(v)

            del extended_cfg_overrides["defaults"]

        cfg = omegaconf.OmegaConf.merge(cfg, extended_cfg_defaults)
        cfg = omegaconf.OmegaConf.merge(cfg, extended_cfg_overrides)

    if overrides is not None:
        cfg = omegaconf.OmegaConf.merge(
            cfg, omegaconf.OmegaConf.from_dotlist(overrides)
        )

    cfg.habitat.task.action_order = omegaconf.ListConfig(action_order)
    omegaconf.OmegaConf.set_struct(cfg, True)
    omegaconf.OmegaConf.set_readonly(cfg, True)

    return cfg
