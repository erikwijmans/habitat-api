#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.RUN_NAME = "debug"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.LOAD_FROM_CKPT = False
_C.RL.CKPT_TO_LOAD = "data/checkpoints/ckpt.0.pth"
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.use_different_critic_lr = False
_C.RL.PPO.critic_lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.use_clipped_value_loss = True
_C.RL.PPO.hidden_size = 512
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"
_C.RL.DDPPO.rnn_type = "LSTM"
_C.RL.DDPPO.num_recurrent_layers = 2
_C.RL.DDPPO.backbone = "resnet50"
_C.RL.DDPPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Loads just the visual encoder backbone weights
_C.RL.DDPPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.DDPPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
# -----------------------------------------------------------------------------
# DAGGER ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.DAGGER = CN()
_C.DAGGER.LR = 1e-3
_C.DAGGER.ITERATIONS = 5
_C.DAGGER.EPOCHS = 10
_C.DAGGER.UPDATE_SIZE = 20000
_C.DAGGER.BATCH_SIZE = 5
_C.DAGGER.P = 0.75
_C.DAGGER.LMDB_MAP_SIZE = 1e9  # 1GB
# How often to commit the writes to the DB,
# less commits is better, but everything must be in memory until a commit happens
_C.DAGGER.LMDB_COMMIT_FREQUENCY = 500
_C.DAGGER.USE_IW = False
# -----------------------------------------------------------------------------
# VLN CONFIG
# -----------------------------------------------------------------------------
_C.VLN = CN()
# on GT trajectories in the training set
_C.VLN.inflection_weight_coef = 3.2
_C.VLN.INSTRUCTION_ENCODER = CN()
_C.VLN.INSTRUCTION_ENCODER.vocab_size = 5000
_C.VLN.INSTRUCTION_ENCODER.max_length = 200
_C.VLN.INSTRUCTION_ENCODER.use_pretrained_embeddings = False
_C.VLN.INSTRUCTION_ENCODER.embedding_file = "data/glove/glove.42B.300d.txt"
_C.VLN.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/vln/r2r/v1/train/train.json.gz"
)
_C.VLN.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.VLN.INSTRUCTION_ENCODER.embedding_size = 200
_C.VLN.INSTRUCTION_ENCODER.hidden_size = 512
_C.VLN.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.VLN.INSTRUCTION_ENCODER.final_state_only = True
_C.VLN.VISUAL_ENCODER = CN()
# TODO create setting for SimpleCNN to process the combined RGB+Depth image.
# VISUAL_ENCODER cnn_type must be of 'SimpleRGBCNN' or 'TorchVisionResNet50'
_C.VLN.VISUAL_ENCODER.cnn_type = "SimpleRGBCNN"
_C.VLN.VISUAL_ENCODER.output_size = 512
# relu or tanh
_C.VLN.VISUAL_ENCODER.activation = "relu"
_C.VLN.DEPTH_ENCODER = CN()
# or VlnResnetDepthEncoder
_C.VLN.DEPTH_ENCODER.cnn_type = "SimpleDepthCNN"
_C.VLN.DEPTH_ENCODER.output_size = 512
# type of resnet to use
_C.VLN.DEPTH_ENCODER.backbone = "NONE"
# path to DDPPO resnet weights
_C.VLN.DEPTH_ENCODER.ddppo_checkpoint = "NONE"
_C.VLN.STATE_ENCODER = CN()
_C.VLN.STATE_ENCODER.hidden_size = 512
_C.VLN.STATE_ENCODER.rnn_type = "GRU"
_C.VLN.RCM = CN()
_C.VLN.RCM.use = False
_C.VLN.RCM.rcm_state_encoder = True
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
