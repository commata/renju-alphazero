"""Residual policy/value network. Forward returns raw logits and current-turn value."""
import torch
from torch import nn

from .config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.layers(x))


class PolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        actions = config.board_size ** 2
        self.trunk = nn.Sequential(
            nn.Conv2d(config.input_planes, config.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels), nn.ReLU(),
            *(ResidualBlock(config.channels) for _ in range(config.blocks)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, config.policy_channels, 1, bias=False),
            nn.BatchNorm2d(config.policy_channels), nn.ReLU(), nn.Flatten(),
            nn.Linear(config.policy_channels * actions, actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(config.channels, config.value_channels, 1, bias=False),
            nn.BatchNorm2d(config.value_channels), nn.ReLU(), nn.Flatten(),
            nn.Linear(config.value_channels * actions, config.value_hidden), nn.ReLU(),
            nn.Linear(config.value_hidden, 1), nn.Tanh(),
        )
        # Kaiming for ReLU convolutions; PyTorch defaults for Linear and BatchNorm.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.config.input_planes, self.config.board_size, self.config.board_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected or x.shape[0] == 0:
            raise ValueError(f"input must have shape [B,{expected}] with B > 0")
        trunk = self.trunk(x)
        return self.policy_head(trunk), self.value_head(trunk)
