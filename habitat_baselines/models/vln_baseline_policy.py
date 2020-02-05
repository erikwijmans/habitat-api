import abc

import torch
from gym import Space

from habitat import Config
from habitat_baselines.models.instruction_encoder import InstructionEncoder
from habitat_baselines.models.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from habitat_baselines.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.models.simple_cnn import (
    SimpleCNN,
    SimpleDepthCNN,
    SimpleRGBCNN,
)
from habitat_baselines.rl.ppo.policy import Net, Policy


class VLNBaselinePolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, vln_config: Config
    ):
        super().__init__(
            VLNBaselineNet(
                observation_space=observation_space, vln_config=vln_config
            ),
            action_space.n,
        )


class VLNBaselineNet(Net):
    r"""A baseline network for vision and language navigation. Simply
    concatenates the encodings of RGB, D, and instruction before decoding
    an action with an RNN.

    Modules:
        Instruction encoder
        Depth encoder
        Visual (RGB) encoder
        RNN state decoder
    """

    def __init__(self, observation_space: Space, vln_config: Config):
        super().__init__()
        self.vln_config = vln_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            vln_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert vln_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if vln_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, vln_config.DEPTH_ENCODER.output_size
            )
        elif vln_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=vln_config.DEPTH_ENCODER.output_size,
                checkpoint=vln_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=vln_config.DEPTH_ENCODER.backbone,
            )

        # Init the RGB visual encoder
        assert vln_config.VISUAL_ENCODER.cnn_type in [
            "SimpleCNN",
            "TorchVisionResNet50",
        ], "VISUAL_ENCODER.cnn_type must be either 'SimpleCNN' or 'TorchVisionResNet50'."

        if vln_config.VISUAL_ENCODER.cnn_type == "SimpleRGBCNN":
            self.visual_encoder = SimpleRGBCNN(
                observation_space, vln_config.VISUAL_ENCODER.output_size
            )
        elif vln_config.VISUAL_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", vln_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.visual_encoder = TorchVisionResNet50(
                observation_space,
                vln_config.VISUAL_ENCODER.output_size,
                device,
                activation=vln_config.VISUAL_ENCODER.activation,
            )

        # Init the RNN state decoder
        rnn_input_size = self.instruction_encoder.output_size
        rnn_input_size += vln_config.DEPTH_ENCODER.output_size
        rnn_input_size += vln_config.VISUAL_ENCODER.output_size

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=vln_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=vln_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.vln_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x VISUAL_ENCODER.output_size]
        """
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.visual_encoder(observations)

        x = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=1
        )
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states
