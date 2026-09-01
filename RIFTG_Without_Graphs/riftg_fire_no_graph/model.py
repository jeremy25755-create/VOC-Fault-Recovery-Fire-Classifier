"""Graph-free ternary classifier for the indoor-fire sensor windows."""

from __future__ import annotations

from torch import Tensor, nn

from .encoder import GraphTemporalEncoder, canonicalize_sensor_windows


class EncodedFeatureClassifier(nn.Module):
    """Map the concatenated per-sensor embeddings to three class logits."""

    def __init__(self, n_nodes: int, hidden_dim: int, n_classes: int = 3) -> None:
        super().__init__()
        self.fully_connected = nn.Linear(n_nodes * hidden_dim, n_classes)

    def forward(self, encoded: Tensor) -> Tensor:
        return self.fully_connected(encoded.flatten(start_dim=1))


class FireRIFTGNoGraphClassifier(nn.Module):
    """Shared sensor encoder followed directly by the original FC classifier.

    This graph-free ablation deliberately contains no adjacency matrix, graph
    fusion, message passing, or RIFTG stability update. Keeping the encoder and
    final classifier unchanged isolates the contribution of graph processing.
    """

    def __init__(
        self,
        n_nodes: int = 14,
        window_length: int = 60,
        hidden_dim: int = 120,
        n_classes: int = 3,
        encoder_mode: str = "current-mixer",
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.encoder = GraphTemporalEncoder(
            n_nodes=n_nodes,
            window_length=window_length,
            hidden_dim=hidden_dim,
            mode=encoder_mode,
        )
        self.classifier = EncodedFeatureClassifier(n_nodes, hidden_dim, n_classes)

    def forward(
        self,
        windows: Tensor,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        windows = canonicalize_sensor_windows(
            windows,
            n_nodes=self.n_nodes,
            window_length=self.window_length,
        )
        encoded = self.encoder(windows)
        logits = self.classifier(encoded)
        if not return_diagnostics:
            return logits
        return logits, {
            "encoded_features": encoded,
            "classifier_features": encoded,
        }
