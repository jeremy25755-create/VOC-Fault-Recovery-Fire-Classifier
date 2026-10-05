"""Clean-only trend recovery graph and fault detector for frozen no-graph models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def causal_moving_average(values: Tensor, kernel_size: int) -> Tensor:
    """Causal moving average with replicated left padding."""
    if values.ndim < 2:
        raise ValueError("values must end with a time dimension")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    flattened = values.reshape(-1, 1, values.shape[-1])
    padded = F.pad(flattened, (kernel_size - 1, 0), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).reshape(values.shape)


class VOCTrendRecoveryGraph(nn.Module):
    """Directed multi-head messages from 13 observed sensors to masked VOC."""

    def __init__(
        self,
        source_count: int,
        window_length: int = 60,
        hidden_dim: int = 64,
        heads: int = 4,
        short_kernel: int = 9,
        long_kernel: int = 21,
    ) -> None:
        super().__init__()
        self.source_count = int(source_count)
        self.window_length = int(window_length)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.short_kernel = int(short_kernel)
        self.long_kernel = int(long_kernel)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(window_length * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.sensor_embedding = nn.Parameter(torch.empty(source_count, hidden_dim))
        self.voc_query = nn.Parameter(torch.empty(hidden_dim))
        self.edge_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, heads),
        )
        self.edge_value = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * heads + hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, window_length * 2),
        )
        nn.init.normal_(self.sensor_embedding, std=0.02)
        nn.init.normal_(self.voc_query, std=0.02)

    def target_trend(self, voc_window: Tensor) -> Tensor:
        return causal_moving_average(voc_window, self.long_kernel)

    def source_features(self, sources: Tensor) -> Tensor:
        short = causal_moving_average(sources, self.short_kernel)
        long = causal_moving_average(sources, self.long_kernel)
        return torch.cat((sources, short, long), dim=-1)

    def forward(self, sources: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if sources.ndim != 3 or sources.shape[1:] != (self.source_count, self.window_length):
            raise ValueError("sources must have shape [batch, 13, 60]")
        batch = len(sources)
        hidden = self.temporal_encoder(self.source_features(sources))
        hidden = hidden + self.sensor_embedding.unsqueeze(0)
        query = self.voc_query.view(1, 1, -1).expand(batch, self.source_count, -1)
        logits = self.edge_score(torch.cat((hidden, query), dim=-1))
        attention = torch.softmax(logits.transpose(1, 2), dim=-1)
        messages = torch.einsum("bhn,bnd->bhd", attention, self.edge_value(hidden))
        decoder_input = torch.cat(
            (messages.flatten(start_dim=1), self.voc_query.unsqueeze(0).expand(batch, -1)),
            dim=-1,
        )
        mean, raw_scale = self.decoder(decoder_input).chunk(2, dim=-1)
        return mean, F.softplus(raw_scale).clamp(1e-3, 1.0), attention


class NoGraphVOCRecoveryClassifier(nn.Module):
    """Exact clean bypass plus selective trend replacement for VOC faults."""

    def __init__(
        self,
        base_model: nn.Module,
        voc_index: int,
        source_indices: Tensor,
        window_length: int = 60,
        recovery_hidden_dim: int = 64,
        recovery_heads: int = 4,
        short_kernel: int = 9,
        long_kernel: int = 21,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.voc_index = int(voc_index)
        sources = torch.as_tensor(source_indices, dtype=torch.long).reshape(-1)
        self.register_buffer("source_indices", sources)
        self.recovery = VOCTrendRecoveryGraph(
            source_count=len(sources),
            window_length=window_length,
            hidden_dim=recovery_hidden_dim,
            heads=recovery_heads,
            short_kernel=short_kernel,
            long_kernel=long_kernel,
        )
        self.register_buffer("relation_threshold", torch.ones(()))
        self.register_buffer("point_threshold", torch.ones(()))
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()

    def freeze_recovery(self) -> None:
        self.recovery.requires_grad_(False)
        self.recovery.eval()

    def recovery_parameters(self):
        return self.recovery.parameters()

    def recover(self, windows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.recovery(windows[:, self.source_indices, :])

    def raw_fault_scores(
        self,
        windows: Tensor,
        recovered_mean: Tensor,
        recovered_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        voc = windows[:, self.voc_index, :]
        observed_trend = self.recovery.target_trend(voc)
        standardized = (observed_trend - recovered_mean).abs() / (recovered_scale + 1e-6)
        high_frequency = voc - causal_moving_average(voc, self.recovery.short_kernel)
        relation_score = standardized.median(dim=-1).values
        point_score = torch.sqrt(high_frequency.square().mean(dim=-1) + 1e-12)
        return relation_score, point_score

    @torch.no_grad()
    def set_thresholds(self, relation: float, point: float) -> None:
        relation, point = float(relation), float(point)
        if relation <= 0 or point <= 0:
            raise ValueError("fault thresholds must be positive")
        self.relation_threshold.fill_(relation)
        self.point_threshold.fill_(point)

    def detect(self, relation_score: Tensor, point_score: Tensor) -> tuple[Tensor, Tensor]:
        relation_ratio = relation_score / self.relation_threshold.clamp_min(1e-8)
        point_ratio = point_score / self.point_threshold.clamp_min(1e-8)
        fault_ratio = torch.maximum(relation_ratio, point_ratio)
        return fault_ratio > 1.0, fault_ratio

    def repaired_windows(self, windows: Tensor, recovered_mean: Tensor) -> Tensor:
        repaired = windows.clone()
        repaired[:, self.voc_index, :] = recovered_mean
        return repaired

    def forward(
        self,
        windows: Tensor,
        return_diagnostics: bool = False,
        force_fault: bool | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        clean_logits = self.base(windows)
        recovered_mean, recovered_scale, attention = self.recover(windows)
        repaired = self.repaired_windows(windows, recovered_mean)
        repaired_logits = self.base(repaired)
        relation_score, point_score = self.raw_fault_scores(
            windows, recovered_mean, recovered_scale
        )
        detected, fault_ratio = self.detect(relation_score, point_score)
        if force_fault is not None:
            detected = torch.full_like(detected, bool(force_fault))
        logits = torch.where(detected.unsqueeze(-1), repaired_logits, clean_logits)
        if not return_diagnostics:
            return logits
        return logits, {
            "detected_fault": detected,
            "fault_ratio": fault_ratio,
            "relation_score": relation_score,
            "point_score": point_score,
            "recovered_voc_trend": recovered_mean,
            "recovered_scale": recovered_scale,
            "recovery_attention": attention,
            "repaired_windows": repaired,
            "clean_logits": clean_logits,
            "repaired_logits": repaired_logits,
        }
