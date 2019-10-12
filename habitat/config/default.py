#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import os.path as osp
from typing import List, Optional, Union

import omegaconf

from habitat.core.utils import Singleton


class _HabitatBaseConfigSearcher(metaclass=Singleton):
    config_paths: List[str] = []
    search_paths: List[str] = []
    recursive_search_paths: List[str] = []

    @classmethod
    def add_config_path(cls, path: str):
        assert osp.exists(path), f'"{path}" does not exist'
        cls.config_paths.append(path)

    @classmethod
    def add_search_path(cls, path: str):
        assert osp.exists(path), f'"{path}" does not exist'
        cls.search_paths.append(path)

    @classmethod
    def add_recursive_search_path(cls, path: str):
        assert osp.exists(path), f'"{path}" does not exist'
        cls.recursive_search_paths.append(path)

    @classmethod
    def build_cfg(cls) -> omegaconf.OmegaConf:
        cfg = omegaconf.OmegaConf.create()

        for cfg_file in cls.config_paths:
            cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.load(cfg))

        for search_path in cls.search_paths:
            for f in glob.glob(osp.join(search_path, "*.yaml")):
                cfg = omegaconf.OmegaConf.merge(
                    cfg, omegaconf.OmegaConf.load(f)
                )

        for search_path in cls.recursive_search_paths:
            for f in glob.glob(
                osp.join(search_path, "**/*.yaml"), recursive=True
            ):
                cfg = omegaconf.OmegaConf.merge(
                    cfg, omegaconf.OmegaConf.load(f)
                )

        return cfg


HabitatBaseConfigSearcher = _HabitatBaseConfigSearcher()

HabitatBaseConfigSearcher.add_recursive_search_path(
    osp.join(osp.abspath(osp.dirname(__file__)), "base")
)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> omegaconf.OmegaConf:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """

    cfg = HabitatBaseConfigSearcher.build_cfg()

    if config_paths is not None:
        if isinstance(config_paths, str):
            config_paths = config_paths.split(",")

        for path in config_paths:
            cfg = omegaconf.OmegaConf.merge(
                cfg, omegaconf.OmegaConf.load(path)
            )

    if opts is not None:
        cfg = omegaconf.OmegaConf.merge(
            cfg, omegaconf.OmegaConf.from_dotlist(ops)
        )

    return cfg
