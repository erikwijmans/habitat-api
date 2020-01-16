#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import inspect
import os
import os.path as osp
from typing import List, Optional, Union

import attr
import hydra
import hydra.experimental
import omegaconf
from hydra._internal.hydra import GlobalHydra

from habitat.core.utils import Singleton


@attr.s(auto_attribs=True)
class HabitatHydraLoader(metaclass=Singleton):
    provider_names: List[str] = ["habitat"]
    provider_paths: List[str] = ["pkg://habitat.config.base"]
    base_configs: List[str] = ["habitat_base.yaml"]

    def _load_file(self, config_file: str):
        extended_cfg_defaults = hydra.experimental.compose(
            config_file, overrides=[]
        )
        omegaconf.OmegaConf.set_struct(extended_cfg_defaults, True)

        extended_cfg_overrides = omegaconf.OmegaConf.load(config_file)
        if "defaults" in extended_cfg_overrides:
            del extended_cfg_overrides["defaults"]

        return omegaconf.OmegaConf.merge(
            extended_cfg_defaults, extended_cfg_overrides
        )

    def get_config(
        self,
        config_paths: Optional[Union[List[str], str]] = None,
        overrides: Optional[Union[List[str], str]] = None,
    ):
        if not GlobalHydra().is_initialized():
            stack = inspect.stack()
            hydra.experimental.initialize(caller_stack_depth=len(stack))

        for name, path in zip(self.provider_names, self.provider_paths):
            if not any(
                path.provider == name
                for path in GlobalHydra().hydra.config_loader.config_search_path.config_search_path
            ):
                GlobalHydra().hydra.config_loader.config_search_path.prepend(
                    name, path
                )

        cfg = omegaconf.OmegaConf.merge(
            *[
                hydra.experimental.compose(base_cfg)
                for base_cfg in self.base_configs
            ]
        )

        if config_paths is not None:
            if isinstance(config_paths, str):
                config_paths = [config_paths]

            cfg = omegaconf.OmegaConf.merge(
                cfg, *[self._load_file(cfg_file) for cfg_file in config_paths]
            )

        omegaconf.OmegaConf.set_struct(cfg, True)
        if overrides is not None:
            if isinstance(overrides, str):
                overrides = [overrides]

            overrides_cfg = hydra.experimental.compose(None, overrides)
            cfg = omegaconf.OmegaConf.merge(cfg, overrides_cfg)

        omegaconf.OmegaConf.set_readonly(cfg, True)
        return cfg


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    overrides: Optional[Union[List[str], str]] = None,
):
    return HabitatHydraLoader().get_config(config_paths, overrides)
