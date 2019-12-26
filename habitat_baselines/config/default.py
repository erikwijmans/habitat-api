#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import inspect
import os
import os.path as osp
from typing import List, Optional, Union

import hydra
import hydra.experimental
import omegaconf
from hydra._internal.hydra import GlobalHydra

from habitat.config import get_config as get_hab_config


def _load_file(config_file: str):
    extended_cfg_defaults = hydra.experimental.compose(
        config_file, overrides=[]
    )
    extended_cfg_overrides = omegaconf.OmegaConf.load(config_file)
    if "defaults" in extended_cfg_overrides:
        del extended_cfg_overrides["defaults"]

    return omegaconf.OmegaConf.merge(
        extended_cfg_defaults, extended_cfg_overrides
    )


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    overrides: Optional[Union[List[str], str]] = None,
):
    if not GlobalHydra().is_initialized():
        stack = inspect.stack()
        hydra.experimental.initialize(caller_stack_depth=len(stack))

    if not any(
        path.provider == "habitat_baselines"
        for path in GlobalHydra().hydra.config_loader.config_search_path.config_search_path
    ):
        GlobalHydra().hydra.config_loader.config_search_path.prepend(
            "habitat_baselines", "pkg://habitat_baselines.config.base"
        )

    cfg = get_hab_config()

    with omegaconf.open_dict(cfg), omegaconf.read_write(cfg):
        cfg = omegaconf.OmegaConf.merge(
            cfg, hydra.experimental.compose("habitat_baselines_base.yaml", [])
        )
        if config_paths is not None:
            if isinstance(config_paths, str):
                config_paths = [config_paths]

            cfg = omegaconf.OmegaConf.merge(
                cfg, *[_load_file(cfg_file) for cfg_file in config_paths]
            )

        if overrides is not None:
            if isinstance(overrides, str):
                overrides = [overrides]

            overrides_cfg = hydra.experimental.compose(None, overrides)
            cfg = omegaconf.OmegaConf.merge(cfg, overrides_cfg)

    omegaconf.OmegaConf.set_struct(cfg, True)
    omegaconf.OmegaConf.set_readonly(cfg, True)

    return cfg
