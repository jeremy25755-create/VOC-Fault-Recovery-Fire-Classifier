"""Clean-only trend recovery graph and fault-mode adapter for frozen no-graph models."""

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
    original_shape = values.shape
    flattened = values.reshape(-1, 1, original_shape[-1])
    padded = F.pad(flattened, (kernel_size - 1, 0), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return smoothed.reshape(original_shape)


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
        self.sensor_embedding = nn.Parameter(
            torch.empty(source_count, hidden_dim)
        )
        self.voc_query = nn.Parameter(torch.empty(hidden_dim))
        self.edge_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, heads),
        )
        self.edge_value = nn.Linear(hidden_dim, hidden_dim)
        decoder_input = hidden_dim * heads + hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim * 4),
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
        if sources.ndim != 3 or sources.shape[1:] != (
            self.source_count,
            self.window_length,
        ):
            raise ValueError("sources must have shape [batch, 13, 60]")
        hidden = self.temporal_encoder(self.source_features(sources))
        hidden = hidden + self.sensor_embedding.unsqueeze(0)
        query = self.voc_query.view(1, 1, -1).expand(
            len(sources), self.source_count, -1
        )
        logits = self.edge_score(torch.cat((hidden, query), dim=-1))
        attention = torch.softmax(logits.transpose(1, 2), dim=-1)
        values = self.edge_value(hidden)
        messages = torch.einsum("bhn,bnd->bhd", attention, values)
        decoded = self.decoder(
            torch.cat(
                (
                    messages.flatten(start_dim=1),
                    self.voc_query.unsqueeze(0).expand(len(sources), -1),
                ),
                dim=-1,
            )
        )
        mean, raw_scale = decoded.chunk(2, dim=-1)
        scale = F.softplus(raw_scale).clamp(1e-3, 1.0)
        return mean, scale, attention


class FaultModeAdapter(nn.Module):
    """Small residual-logit adapter used only after confirmed VOC replacement."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, encoded: Tensor) -> Tensor:
        return self.network(encoded.flatten(start_dim=1))


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
        adapter_hidden_dim: int = 32,
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
        self.adapter = FaultModeAdapter(
            input_dim=base_model.n_nodes * base_model.hidden_dim,
            hidden_dim=adapter_hidden_dim,
            classes=base_model.n_classes,
        )
        self.register_buffer("relation_threshold", torch.ones(base_model.n_classes))
        self.register_buffer("point_threshold", torch.ones(base_model.n_classes))
        self.freeze_base()

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

    def freeze_recovery(self) -> None:
        for parameter in self.recovery.parameters():
            parameter.requires_grad = False
        self.recovery.eval()

    def recovery_parameters(self):
        return self.recovery.parameters()

    def adapter_parameters(self):
        return self.adapter.parameters()

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
        standardized = (observed_trend - recovered_mean).abs() / (
            recovered_scale + 1e-6
        )
        relation_score = standardized.median(dim=-1).values
        short = causal_moving_average(voc, self.recovery.short_kernel)
        high_frequency = voc - short
        point_score = torch.sqrt(high_frequency.square().mean(dim=-1) + 1e-12)
        return relation_score, point_score

    @torch.no_grad()
    def set_thresholds(self, relation: Tensor, point: Tensor) -> None:
        relation_value = torch.as_tensor(
            relation,
            dtype=self.relation_threshold.dtype,
            device=self.relation_threshold.device,
        )
        point_value = torch.as_tensor(
            point,
            dtype=self.point_threshold.dtype,
            device=self.point_threshold.device,
        )
        relation_value = relation_value.reshape_as(self.relation_threshold)
        point_value = point_value.reshape_as(self.point_threshold)
        if torch.any(relation_value <= 0) or torch.any(point_value <= 0):
            raise ValueError("fault thresholds must be positive")
        self.relation_threshold.copy_(relation_value)
        self.point_threshold.copy_(point_value)

    def detect(
        self,
        relation_score: Tensor,
        point_score: Tensor,
        recovery_mode: Tensor,
    ) -> tuple[Tensor, Tensor]:
        relation_threshold = self.relation_threshold[recovery_mode]
        point_threshold = self.point_threshold[recovery_mode]
        relation_ratio = relation_score / relation_threshold.clamp_min(1e-8)
        point_ratio = point_score / point_threshold.clamp_min(1e-8)
        fault_ratio = torch.maximum(relation_ratio, point_ratio)
        return fault_ratio > 1.0, fault_ratio

    def repaired_windows(self, windows: Tensor, recovered_mean: Tensor) -> Tensor:
        repaired = windows.clone()
        repaired[:, self.voc_index, :] = recovered_mean
        return repaired

    def fault_logits(
        self,
        repaired_windows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoded = self.base.encoder(repaired_windows)
        base_logits = self.base.classifier(encoded)
        return base_logits + self.adapter(encoded), encoded

    def forward(
        self,
        windows: Tensor,
        return_diagnostics: bool = False,
        force_fault: bool | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        clean_logits = self.base(windows)
        recovered_mean, recovered_scale, attention = self.recover(windows)
        repaired = self.repaired_windows(windows, recovered_mean)
        repaired_logits, repaired_encoded = self.fault_logits(repaired)
        recovery_mode = repaired_logits.argmax(dim=-1)
        relation_score, point_score = self.raw_fault_scores(
            windows, recovered_mean, recovered_scale
        )
        detected, fault_ratio = self.detect(
            relation_score, point_score, recovery_mode
        )
        if force_fault is not None:
            detected = torch.full_like(detected, bool(force_fault))
        logits = torch.where(detected.unsqueeze(-1), repaired_logits, clean_logits)
        if not return_diagnostics:
            return logits
        return logits, {
            "detected_fault": detected,
            "fault_ratio": fault_ratio,
            "recovery_mode": recovery_mode,
            "relation_score": relation_score,
            "point_score": point_score,
            "recovered_voc_trend": recovered_mean,
            "recovered_scale": recovered_scale,
            "recovery_attention": attention,
            "repaired_windows": repaired,
            "repaired_encoded": repaired_encoded,
            "clean_logits": clean_logits,
            "repaired_logits": repaired_logits,
        }
