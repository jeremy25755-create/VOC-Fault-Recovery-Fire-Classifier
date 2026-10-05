"""Train the graph-free RIFTG indoor-fire ablation classifier."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent

from voc_fire.data import (
    FEATURE_COLUMNS,
    INDEX_TO_LABEL,
    ClassificationDatasetBundle,
    make_loaders,
    prepare_dataset,
)
from voc_fire.metrics import classification_metrics
from voc_fire.model import FireRIFTGNoGraphClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "Indoor_Fire_Except_Sensor0011.csv",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=120)
    parser.add_argument(
        "--encoder-mode",
        choices=("linear", "current-mixer"),
        default="current-mixer",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=("fraction", "date"),
        default="date",
    )
    parser.add_argument("--train-start-date", default="2022-07-04")
    parser.add_argument("--train-end-date", default="2022-07-06")
    parser.add_argument("--validation-date", default="2022-07-07")
    parser.add_argument("--test-date", default="2022-07-08")
    parser.add_argument("--max-gap-seconds", type=float, default=30.0)
    parser.add_argument(
        "--balanced-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "except0011_no_graph_date_split_3seeds",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def runtime_environment(device_name: str) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device_name,
        "device_name": (
            torch.cuda.get_device_name(torch.device(device_name))
            if device_name.startswith("cuda") and torch.cuda.is_available()
            else platform.processor() or "CPU"
        ),
    }


@torch.no_grad()
def predict(
    model: FireRIFTGNoGraphClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        logits = model(windows)
        probabilities = torch.softmax(logits, dim=-1)
        actual_parts.append(labels.numpy())
        predicted_parts.append(probabilities.argmax(dim=-1).cpu().numpy())
        probability_parts.append(probabilities.cpu().numpy())
    return (
        np.concatenate(actual_parts),
        np.concatenate(predicted_parts),
        np.concatenate(probability_parts),
    )


def build_model(args: argparse.Namespace) -> FireRIFTGNoGraphClassifier:
    return FireRIFTGNoGraphClassifier(
        n_nodes=len(FEATURE_COLUMNS),
        window_length=args.window_length,
        hidden_dim=args.hidden_size,
        n_classes=3,
        encoder_mode=args.encoder_mode,
    )


def plot_run(
    run_dir: Path,
    history: list[dict[str, float]],
    metrics: dict[str, object],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    epochs = [entry["epoch"] for entry in history]
    axes[0, 0].plot(epochs, [entry["train_loss"] for entry in history], marker="o")
    axes[0, 0].set(title="Training loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0, 1].plot(
        epochs,
        [entry["validation_macro_f1"] for entry in history],
        marker="o",
        color="darkorange",
    )
    axes[0, 1].set(title="Validation", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    image = axes[1, 0].imshow(matrix, cmap="Blues")
    class_names = [INDEX_TO_LABEL[index] for index in range(3)]
    axes[1, 0].set(
        title="Test confusion matrix",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=range(3),
        yticks=range(3),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    for row in range(3):
        for column in range(3):
            axes[1, 0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)
    per_class = metrics["per_class"]
    axes[1, 1].bar(
        class_names,
        [per_class[name]["f1"] for name in class_names],
        color=["slategray", "firebrick", "darkorange"],
    )
    axes[1, 1].set(title="Test F1 by class", ylabel="F1", ylim=(0, 1))
    figure.savefig(run_dir / "diagnostics.png", dpi=180)
    plt.close(figure)


def train_one_run(
    args: argparse.Namespace,
    bundle: ClassificationDatasetBundle,
    run_index: int,
) -> dict[str, object]:
    run_seed = args.seed + run_index
    set_seed(run_seed, args.deterministic)
    train_loader, validation_loader, test_loader = make_loaders(
        bundle,
        batch_size=args.batch_size,
        seed=run_seed,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    model = build_model(args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    weight = (
        torch.as_tensor(bundle.class_weights, dtype=torch.float32, device=device)
        if args.balanced_class_weights
        else None
    )
    loss_function = nn.CrossEntropyLoss(weight=weight)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_f1 = -1.0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for windows, labels in train_loader:
            windows = windows.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(windows)
            seen += len(windows)

        val_actual, val_predicted, _ = predict(model, validation_loader, device)
        val_metrics = classification_metrics(val_actual, val_predicted)
        train_loss = total_loss / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": float(val_metrics["accuracy"]),
                "validation_macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        if float(val_metrics["macro_f1"]) > best_validation_f1:
            best_validation_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[run {run_index + 1:02d}] epoch {epoch:02d}/{args.epochs} "
            f"loss={train_loss:.6f} val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_macro_F1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    actual, predicted, probabilities = predict(model, test_loader, device)
    metrics = classification_metrics(actual, predicted)
    elapsed = time.perf_counter() - started
    run_dir = args.output / f"run_{run_index + 1:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "arguments": vars(args),
            "dataset_metadata": bundle.metadata,
            "feature_columns": FEATURE_COLUMNS,
            "class_names": [INDEX_TO_LABEL[index] for index in range(3)],
            "metrics": metrics,
            "best_validation_macro_f1": best_validation_f1,
            "parameter_count": parameter_count,
            "run_seed": run_seed,
        },
        run_dir / "model.pt",
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        actual=actual,
        predicted=predicted,
        probabilities=probabilities,
    )
    with (run_dir / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)
    plot_run(run_dir, history, metrics)
    print(
        f"[run {run_index + 1:02d}] TEST accuracy={metrics['accuracy']:.4f} "
        f"macro_F1={metrics['macro_f1']:.4f} weighted_F1={metrics['weighted_f1']:.4f}",
        flush=True,
    )
    for name, values in metrics["per_class"].items():
        print(
            f"  {name:<10} F1={values['f1']:.4f} recall={values['recall']:.4f} "
            f"precision={values['precision']:.4f} support={values['support']}",
            flush=True,
        )
    return {
        "run": run_index + 1,
        "seed": run_seed,
        "metrics": metrics,
        "best_validation_macro_f1": best_validation_f1,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed,
        "run_directory": str(run_dir.resolve()),
    }


def summarize(
    args: argparse.Namespace,
    bundle: ClassificationDatasetBundle,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    scalar_metrics = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
    mean = {
        name: float(np.mean([run["metrics"][name] for run in runs]))
        for name in scalar_metrics
    }
    std = {
        name: float(np.std([run["metrics"][name] for run in runs], ddof=0))
        for name in scalar_metrics
    }
    class_names = [INDEX_TO_LABEL[index] for index in range(3)]
    class_metrics = ("f1", "recall", "precision")
    per_class_mean: dict[str, dict[str, float]] = {}
    per_class_std: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        per_class_mean[class_name] = {}
        per_class_std[class_name] = {}
        for metric in class_metrics:
            values = np.asarray(
                [run["metrics"]["per_class"][class_name][metric] for run in runs],
                dtype=np.float64,
            )
            per_class_mean[class_name][metric] = float(values.mean())
            per_class_std[class_name][metric] = float(values.std(ddof=0))
    return {
        "model": "FireRIFTGNoGraphClassifier",
        "task": "ternary classification: Background / Fire / Nuisance",
        "protocol": {
            "arguments": vars(args),
            "architecture": (
                "14 sensor-variable nodes -> shared 60-to-120 encoder -> flatten "
                "14x120 embeddings -> fully connected layer -> 3 logits"
            ),
            "excluded_components": [
                "Static Graph",
                "Dynamic Graph",
                "Static/Dynamic Attention Fusion",
                "adjacency matrix",
                "message passing",
                "RIFTG stability update",
                "Robust Graph",
                "fault mask",
                "fault-specific HMPO shrinkage/message equation",
            ],
            "runtime_environment": runtime_environment(args.device),
        },
        "dataset": bundle.metadata,
        "mean": mean,
        "std": std,
        "per_class_mean": per_class_mean,
        "per_class_std": per_class_std,
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("runs, epochs, and batch size must be positive")
    if args.hidden_size != 120:
        print(
            f"WARNING: requested hidden size {args.hidden_size}; the designed setting is 120.",
            flush=True,
        )
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = prepare_dataset(
        args.data,
        window_length=args.window_length,
        stride=args.stride,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        max_gap_seconds=args.max_gap_seconds,
        split_mode=args.split_mode,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        validation_date=args.validation_date,
        test_date=args.test_date,
    )
    print(
        f"device={args.device} data={args.data} rows={bundle.metadata['rows']} "
        f"windows={bundle.metadata['window_counts']} input=[14,{args.window_length}] "
        f"sensors={bundle.metadata['sensor_ids']} split={bundle.metadata['split_dates']} "
        f"embedding={args.hidden_size} target=ternary_label model=graph_free",
        flush=True,
    )
    runs = [train_one_run(args, bundle, index) for index in range(args.runs)]
    summary = summarize(args, bundle, runs)
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=str)
    print("\nFINAL TEST RESULTS (mean +/- SD)", flush=True)
    for metric in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ):
        print(
            f"{metric:<12} {summary['mean'][metric]:.4f} +/- {summary['std'][metric]:.4f}",
            flush=True,
        )
    print("\nPER-CLASS TEST RESULTS ACROSS RUNS (mean +/- SD, %)", flush=True)
    print(
        f"{'Class':<12}{'F1':>20}{'Recall':>20}{'Precision':>20}",
        flush=True,
    )
    for class_index in range(3):
        class_name = INDEX_TO_LABEL[class_index]
        mean_values = summary["per_class_mean"][class_name]
        std_values = summary["per_class_std"][class_name]
        print(
            f"{class_name:<12}"
            f"{100.0 * mean_values['f1']:8.2f} +/- {100.0 * std_values['f1']:<7.2f}"
            f"{100.0 * mean_values['recall']:8.2f} +/- {100.0 * std_values['recall']:<7.2f}"
            f"{100.0 * mean_values['precision']:8.2f} +/- {100.0 * std_values['precision']:<7.2f}",
            flush=True,
        )
    print(f"Full results: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
