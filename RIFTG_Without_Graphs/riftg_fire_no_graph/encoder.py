"""Sensor-window encoder used before graph message passing."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def canonicalize_sensor_windows(
    windows: Tensor,
    n_nodes: int,
    window_length: int,
) -> Tensor:
    """Return the canonical [batch, nodes, time] representation."""
    if windows.ndim == 2:
        if tuple(windows.shape) == (window_length, n_nodes):
            return windows.transpose(0, 1).unsqueeze(0).contiguous()
        if tuple(windows.shape) == (n_nodes, window_length):
            return windows.unsqueeze(0)
    elif windows.ndim == 3:
        if tuple(windows.shape[1:]) == (n_nodes, window_length):
            return windows
        if tuple(windows.shape[1:]) == (window_length, n_nodes):
            return windows.transpose(1, 2).contiguous()
    raise ValueError("unexpected sensor-window shape")


class GraphTemporalEncoder(nn.Module):
    """Embed each sensor's 60 observations into a 120-dimensional vector."""

    def __init__(
        self,
        n_nodes: int,
        window_length: int = 60,
        hidden_dim: int = 120,
        mode: str = "current-mixer",
    ) -> None:
        super().__init__()
        if mode not in {"linear", "current-mixer"}:
            raise ValueError("mode must be 'linear' or 'current-mixer'")
        if mode == "current-mixer" and hidden_dim < 2:
            raise ValueError("current-mixer requires hidden_dim >= 2")
        self.n_nodes = n_nodes
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.mode = mode
        projected_dim = hidden_dim if mode == "linear" else hidden_dim - 1
        self.temporal_projection = nn.Linear(window_length, projected_dim)
        self.mixer = (
            nn.Linear(projected_dim, projected_dim)
            if mode == "current-mixer"
            else None
        )
        nn.init.xavier_uniform_(self.temporal_projection.weight)
        nn.init.zeros_(self.temporal_projection.bias)
        if self.mixer is not None:
            nn.init.xavier_uniform_(self.mixer.weight)
            nn.init.zeros_(self.mixer.bias)

    def forward(self, windows: Tensor) -> Tensor:
        if windows.ndim != 3 or windows.shape[1:] != (
            self.n_nodes,
            self.window_length,
        ):
            raise ValueError("windows must have shape [batch, nodes, time]")
        temporal = self.temporal_projection(windows)
        if self.mode == "linear":
            return temporal
        assert self.mixer is not None
        temporal = torch.tanh(self.mixer(torch.tanh(temporal)))
        current = windows[:, :, -1:].contiguous()
        return torch.cat((current, temporal), dim=-1)
