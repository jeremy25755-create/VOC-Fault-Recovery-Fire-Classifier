"""Check whether VOC can be recovered from the other 13 clean sensors."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent

from voc_fire.data import (
    FEATURE_COLUMNS,
    INDEX_TO_LABEL,
    prepare_dataset,
)


class VOCGraphReconstructor(nn.Module):
    """Star-graph message passing from 13 observed sensors to masked VOC."""

    def __init__(
        self,
        source_count: int,
        window_length: int,
        hidden_dim: int = 64,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.source_count = source_count
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.temporal_encoder = nn.Sequential(
            nn.Linear(window_length, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.sensor_embedding = nn.Parameter(
            torch.empty(source_count, hidden_dim)
        )
        self.target_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.edge_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, heads),
        )
        self.edge_value = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * heads + hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, window_length),
        )
        nn.init.normal_(self.sensor_embedding, std=0.02)
        nn.init.normal_(self.target_embedding, std=0.02)

    def forward(self, sources: Tensor) -> tuple[Tensor, Tensor]:
        if sources.ndim != 3 or sources.shape[1:] != (
            self.source_count,
            self.window_length,
        ):
            raise ValueError("sources must have shape [batch, 13, window]")
        hidden = self.temporal_encoder(sources)
        hidden = hidden + self.sensor_embedding.unsqueeze(0)
        target = self.target_embedding.view(1, 1, -1).expand(
            len(sources), self.source_count, -1
        )
        logits = self.edge_score(torch.cat((hidden, target), dim=-1))
        attention = torch.softmax(logits.transpose(1, 2), dim=-1)
        values = self.edge_value(hidden)
        messages = torch.einsum("bhn,bnd->bhd", attention, values)
        decoded = self.decoder(
            torch.cat(
                (
                    messages.flatten(start_dim=1),
                    self.target_embedding.unsqueeze(0).expand(len(sources), -1),
                ),
                dim=-1,
            )
        )
        return decoded, attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "Indoor_Fire_Except_Sensor0011.csv",
    )
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-gap-seconds", type=float, default=20.0)
    parser.add_argument("--train-start-date", default="2022-07-04")
    parser.add_argument("--train-end-date", default="2022-07-06")
    parser.add_argument("--validation-date", default="2022-07-07")
    parser.add_argument("--test-date", default="2022-07-08")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-sample-step", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--trend-loss-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "voc_recovery_r2_diagnostic",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def reconstruction_loss(
    predicted: Tensor,
    target: Tensor,
    trend_weight: float,
) -> Tensor:
    level = nn.functional.smooth_l1_loss(predicted, target)
    predicted_delta = predicted[:, 1:] - predicted[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    trend = nn.functional.smooth_l1_loss(predicted_delta, target_delta)
    return level + trend_weight * trend


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    residual = np.square(actual - predicted).sum()
    total = np.square(actual - actual.mean()).sum()
    return float(1.0 - residual / max(total, 1e-12))


def regression_metrics(
    actual_scaled: np.ndarray,
    predicted_scaled: np.ndarray,
    labels: np.ndarray,
    voc_minimum: float,
    voc_span: float,
) -> dict[str, object]:
    actual = actual_scaled * voc_span + voc_minimum
    predicted = predicted_scaled * voc_span + voc_minimum

    def calculate(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
        difference = y_hat - y
        return {
            "r2_all_points": r2_score(y, y_hat),
            "r2_last_point": r2_score(y[:, -1], y_hat[:, -1]),
            "r2_window_mean": r2_score(y.mean(axis=1), y_hat.mean(axis=1)),
            "mae_raw": float(np.abs(difference).mean()),
            "rmse_raw": float(np.sqrt(np.square(difference).mean())),
            "mae_last_point_raw": float(np.abs(difference[:, -1]).mean()),
            "rmse_last_point_raw": float(
                np.sqrt(np.square(difference[:, -1]).mean())
            ),
        }

    result: dict[str, object] = {"overall": calculate(actual, predicted)}
    result["per_class"] = {}
    for class_index in range(3):
        mask = labels == class_index
        name = INDEX_TO_LABEL[class_index]
        result["per_class"][name] = {
            **calculate(actual[mask], predicted[mask]),
            "windows": int(mask.sum()),
        }
    return result


@torch.inference_mode()
def evaluate(
    model: VOCGraphReconstructor,
    loader: DataLoader,
    source_indices: list[int],
    voc_index: int,
    device: torch.device,
    voc_minimum: float,
    voc_span: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model.eval()
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    attention_sum = np.zeros((model.heads, len(source_indices)), dtype=np.float64)
    seen = 0
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        predicted, attention = model(windows[:, source_indices, :])
        actual_parts.append(windows[:, voc_index, :].cpu().numpy())
        predicted_parts.append(predicted.cpu().numpy())
        label_parts.append(labels.numpy())
        attention_sum += attention.sum(dim=0).cpu().numpy()
        seen += len(windows)
    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    labels = np.concatenate(label_parts)
    metrics = regression_metrics(
        actual, predicted, labels, voc_minimum, voc_span
    )
    diagnostics = {
        "actual_scaled": actual,
        "predicted_scaled": predicted,
        "labels": labels,
        "attention_mean": attention_sum / max(seen, 1),
    }
    return metrics, diagnostics


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    print(
        f"{name} overall R2(all/last/mean)="
        f"{overall['r2_all_points']:.4f}/"
        f"{overall['r2_last_point']:.4f}/"
        f"{overall['r2_window_mean']:.4f} "
        f"MAE={overall['mae_raw']:.4f} RMSE={overall['rmse_raw']:.4f}",
        flush=True,
    )
    for class_index in range(3):
        class_name = INDEX_TO_LABEL[class_index]
        values = metrics["per_class"][class_name]
        print(
            f"  {class_name:<10} R2(all/last/mean)="
            f"{values['r2_all_points']:.4f}/"
            f"{values['r2_last_point']:.4f}/"
            f"{values['r2_window_mean']:.4f} windows={values['windows']}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.epochs < 1 or args.batch_size < 1 or args.train_sample_step < 1:
        raise ValueError("epochs, batch size, and train sample step must be positive")
    set_seed(args.seed)
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    bundle = prepare_dataset(
        args.data,
        window_length=args.window_length,
        stride=args.stride,
        max_gap_seconds=args.max_gap_seconds,
        split_mode="date",
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        validation_date=args.validation_date,
        test_date=args.test_date,
    )
    voc_index = FEATURE_COLUMNS.index("VOC_Room_RAW")
    source_indices = [index for index in range(len(FEATURE_COLUMNS)) if index != voc_index]
    train_subset = Subset(
        bundle.train, range(0, len(bundle.train), args.train_sample_step)
    )
    train_loader = make_loader(
        train_subset,
        args.batch_size,
        device,
        args.num_workers,
        True,
        args.seed,
    )
    validation_loader = make_loader(
        bundle.validation,
        args.batch_size,
        device,
        args.num_workers,
        False,
        args.seed,
    )
    test_loader = make_loader(
        bundle.test,
        args.batch_size,
        device,
        args.num_workers,
        False,
        args.seed,
    )
    model = VOCGraphReconstructor(
        source_count=len(source_indices),
        window_length=args.window_length,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    voc_minimum = float(bundle.scaler.minimum[voc_index])
    voc_span = float(
        max(bundle.scaler.maximum[voc_index] - bundle.scaler.minimum[voc_index], 1e-8)
    )
    print(
        f"device={device} windows={bundle.metadata['window_counts']} "
        f"train_used={len(train_subset)} target=VOC_Room_RAW target_input=MASKED "
        f"sources=13 parameters={parameter_count}",
        flush=True,
    )
    best_state: dict[str, Tensor] | None = None
    best_validation_r2 = -math.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for windows, _ in train_loader:
            windows = windows.to(device, non_blocking=True)
            target = windows[:, voc_index, :]
            predicted, _ = model(windows[:, source_indices, :])
            loss = reconstruction_loss(
                predicted, target, args.trend_loss_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(windows)
            count += len(windows)
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            source_indices,
            voc_index,
            device,
            voc_minimum,
            voc_span,
        )
        train_loss = total_loss / max(count, 1)
        validation_r2 = float(
            validation_metrics["overall"]["r2_all_points"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_r2_all_points": validation_r2,
                "validation_r2_last_point": float(
                    validation_metrics["overall"]["r2_last_point"]
                ),
            }
        )
        if validation_r2 > best_validation_r2:
            best_validation_r2 = validation_r2
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={train_loss:.6f} "
            f"val_R2_all={validation_r2:.4f} "
            f"val_R2_last={validation_metrics['overall']['r2_last_point']:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("training did not create a best checkpoint")
    model.load_state_dict(best_state)
    validation_metrics, validation_diagnostics = evaluate(
        model,
        validation_loader,
        source_indices,
        voc_index,
        device,
        voc_minimum,
        voc_span,
    )
    test_metrics, test_diagnostics = evaluate(
        model,
        test_loader,
        source_indices,
        voc_index,
        device,
        voc_minimum,
        voc_span,
    )
    source_names = [FEATURE_COLUMNS[index] for index in source_indices]
    attention_average = test_diagnostics["attention_mean"].mean(axis=0)
    attention_ranking = [
        {"sensor": source_names[index], "weight": float(attention_average[index])}
        for index in np.argsort(-attention_average)
    ]
    report = {
        "experiment": "clean-only VOC leave-one-sensor-out graph reconstruction",
        "target_used_as_input": False,
        "noise_or_fault_used_for_training": False,
        "arguments": vars(args),
        "dataset": bundle.metadata,
        "parameter_count": parameter_count,
        "source_sensors": source_names,
        "best_validation_r2": best_validation_r2,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_attention_ranking": attention_ranking,
        "history": history,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "arguments": vars(args),
            "feature_columns": FEATURE_COLUMNS,
            "source_indices": source_indices,
            "voc_index": voc_index,
            "parameter_count": parameter_count,
        },
        args.output / "model.pt",
    )
    np.savez_compressed(
        args.output / "predictions.npz",
        validation_actual_scaled=validation_diagnostics["actual_scaled"],
        validation_predicted_scaled=validation_diagnostics["predicted_scaled"],
        validation_labels=validation_diagnostics["labels"],
        test_actual_scaled=test_diagnostics["actual_scaled"],
        test_predicted_scaled=test_diagnostics["predicted_scaled"],
        test_labels=test_diagnostics["labels"],
        test_attention_mean=test_diagnostics["attention_mean"],
    )
    print("\nFINAL VOC RECOVERY RESULTS", flush=True)
    print_metrics("VALIDATION", validation_metrics)
    print_metrics("TEST", test_metrics)
    print("\nTop recovery edges", flush=True)
    for entry in attention_ranking[:7]:
        print(f"  {entry['sensor']:<24} {entry['weight']:.4f}", flush=True)
    print(f"Full results: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
